"""
Batched camera images of MuJoCo Warp states from the MuJoCo Warp ray tracer, shaded like MuJoCo's OpenGL
renderer as robosuite uses it.
"""
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

import jax.numpy as jnp
import mujoco
from mujoco import mjx

# geom groups robosuite renders: everything but collision geoms
_VISIBLE_GEOM_GROUPS = [1, 2]
# geom group that is not rendered
_HIDDEN_GEOM_GROUP = 5
# stands in for zero material shininess, whose specular term OpenGL applies at full strength
_MIN_SHININESS = 1e-6


def _texels(m, tex):
    """Texels of texture `tex` as rows of channels, a view into `m.tex_data`."""
    n = m.tex_width[tex] * m.tex_height[tex] * m.tex_nchannel[tex]
    return m.tex_data[m.tex_adr[tex]:m.tex_adr[tex] + n].reshape(-1, m.tex_nchannel[tex])


def _linearize_srgb_textures(m):
    """Convert the texels of `m`'s sRGB textures to linear color, as OpenGL samples them."""
    for tex in np.nonzero(m.tex_colorspace == mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)[0]:
        texels = _texels(m, tex)
        c = texels[:, :3] / 255.
        linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        texels[:, :3] = np.round(linear * 255.).astype(np.uint8)


def _mean_texture_color(m, tex):
    """Mean RGB color in [0, 1] of texture `tex`."""
    return _texels(m, tex)[:, :3].mean(axis=0) / 255.


def _texture_is_mapped(m, geom):
    """Whether textures on `geom` are mapped like MuJoCo's OpenGL renderer: meshes with texture coordinates."""
    return m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and m.mesh_texcoordadr[m.geom_dataid[geom]] >= 0


def _geom_alpha(m):
    """Opacity of each geom: its material's alpha, or its own without a material."""
    return np.where(m.geom_matid >= 0, m.mat_rgba[m.geom_matid, 3], m.geom_rgba[:, 3])


def _texture_pass(m, context, graph_mode):
    """
    Model and render context whose images are the texture colors, or white where untextured. Textures the
    ray tracer does not map like OpenGL show their mean color.
    """
    textures = deepcopy(m)
    _linearize_srgb_textures(textures)
    textures.mat_rgba[:, :3] = 1.
    textures.geom_rgba[:, :3] = 1.
    for g in range(m.ngeom):
        mat = m.geom_matid[g]
        tex = m.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB] if mat >= 0 else -1
        if tex >= 0 and not _texture_is_mapped(m, g):
            textures.geom_matid[g] = -1
            textures.geom_rgba[g, :3] = _mean_texture_color(textures, tex)
    textures.mat_emission[:] = 0.
    textures.light_active[:] = False
    textures.vis.headlight.active = 1
    textures.vis.headlight.ambient[:] = 1.
    textures.vis.headlight.diffuse[:] = 0.
    textures.vis.headlight.specular[:] = 0.
    mx = mjx.put_model(textures, impl="warp", graph_mode=graph_mode)
    rc = mjx.create_render_context(
        textures, render_depth=False, use_textures=True, enable_specular=False, enable_per_light_ambient=False,
        **context
    )
    return mx, rc


class _Shading:
    """
    OpenGL's color of each pixel's frontmost geom in one scene. OpenGL modulates textures by the clamped lit
    color, so the color is the product of a lit pass without textures and an unlit pass of the textures alone.
    """
    def __init__(self, m, context, depths, segmentation, graph_mode):
        lit = deepcopy(m)
        lit.mat_shininess[:] = np.maximum(lit.mat_shininess, _MIN_SHININESS)
        self.lit_model = mjx.put_model(lit, impl="warp", graph_mode=graph_mode)
        self.lit_context = mjx.create_render_context(
            lit, render_depth=depths, render_seg=segmentation, use_textures=False, **context
        )
        self.texture_model, self.texture_context = _texture_pass(m, context, graph_mode)
        self.segmentation = segmentation

    def __call__(self, d):
        """Lit, texture, depth, and segmentation buffers of the world `d`."""
        rc = self.lit_context.pytree()
        lit_data = mjx.refit_bvh(self.lit_model, d, rc)
        if self.segmentation:
            lit, depth, seg, _ = mjx.render_with_segmentation(self.lit_model, lit_data, rc)
        else:
            (lit, depth, _), seg = mjx.render(self.lit_model, lit_data, rc), None
        rc = self.texture_context.pytree()
        textures, _, _ = mjx.render(self.texture_model, mjx.refit_bvh(self.texture_model, d, rc), rc)
        # an unbatched call returns the buffers of every world, the first of which is `d`'s
        if lit.ndim == 2:
            lit, textures = lit[0], textures[0]
            depth = depth[0] if depth.ndim == 2 else depth
            seg = seg[0] if seg is not None else None
        return lit, textures, depth, seg

    def rgb(self, camera, lit, textures):
        """RGB image in [0, 1] of camera index `camera` from the lit and texture buffers."""
        lit = mjx.get_rgb(self.lit_context.pytree(), camera, lit)
        return lit * mjx.get_rgb(self.texture_context.pytree(), camera, textures)


class CameraRenderer:
    """
    RGB and depth images of named cameras, shaded like MuJoCo's OpenGL renderer. Translucent geoms are blended
    over the scene behind them with their alpha, one layer deep, and occlude like opaque geoms in depth.
    """
    def __init__(
        self, m, camera_names: Sequence[str], height: int, width: int, depth: bool, nworld: int, graph_mode
    ):
        """Calls are unbatched or mapped over exactly `nworld` worlds."""
        self._camera_names = list(camera_names)
        self._height, self._width, self._depth = height, width, depth
        self._far = float(m.vis.map.zfar * m.stat.extent)
        n = len(self._camera_names)
        context = dict(
            nworld=nworld, cam_res=[(width, height)] * n, cam_active=self._camera_names,
            render_rgb=True, use_shadows=False, enabled_geom_groups=_VISIBLE_GEOM_GROUPS,
        )
        alpha = _geom_alpha(m)
        translucent = (alpha < 1.) & np.isin(m.geom_group, _VISIBLE_GEOM_GROUPS)
        blend = bool(translucent.any())
        self._scene = _Shading(m, context, [depth] * n, blend, graph_mode)
        self._behind = self._alpha = None
        if blend:
            behind = deepcopy(m)
            behind.geom_group[translucent] = _HIDDEN_GEOM_GROUP
            self._behind = _Shading(behind, context, False, False, graph_mode)
            self._alpha = jnp.asarray(np.where(translucent, alpha, 1.), jnp.float32)

    def __call__(self, d):
        """
        Images of the world `d`: uint8 RGB images of shape (height, width, 3) under "<camera>_image" and planar
        depth in meters of shape (height, width, 1) under "<camera>_depth", both upright.
        """
        lit, textures, depth, seg = self._scene(d)
        behind = self._behind(d) if self._behind is not None else None
        rc = self._scene.lit_context.pytree()
        obs = {}
        for i, name in enumerate(self._camera_names):
            rgb = self._scene.rgb(i, lit, textures)
            if behind is not None:
                behind_lit, behind_textures, _, _ = behind
                front = mjx.get_segmentation(rc, i, seg)
                alpha = jnp.where(front >= 0, self._alpha[jnp.maximum(front, 0)], 1.)[..., None]
                rgb = alpha * rgb + (1. - alpha) * self._behind.rgb(i, behind_lit, behind_textures)
            obs[f"{name}_image"] = jnp.round(rgb * 255.).astype(jnp.uint8)
            if self._depth:
                z = mjx.get_depth(rc, i, depth, self._far) * self._far
                obs[f"{name}_depth"] = jnp.where(z > 0., z, self._far)
        return obs
