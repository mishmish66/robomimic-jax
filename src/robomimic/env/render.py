"""
Batched camera images of MuJoCo Warp states from the MuJoCo Warp ray tracer, lit and textured in one pass: shadows
from the scene's lights under a dim headlight, antialiased by supersampling, then a camera's defocus, vignetting,
white balance and sensor noise; robosuite's end-effector target markers are not rendered.
"""
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# geom groups robosuite renders: everything but collision geoms
_VISIBLE_GEOM_GROUPS = [1, 2]
# geom group that is not rendered
_HIDDEN_GEOM_GROUP = 5
# stands in for zero material shininess, whose specular term OpenGL applies at full strength
_MIN_SHININESS = 1e-6
# version of the images this module renders, changed with their look
RENDER_VERSION = 2
# rendered pixels per image pixel along each axis, averaged down
SUPERSAMPLE = 2
# whether lights cast shadows
SHADOWS = True
# headlight diffuse and ambient intensity under the scene's own lights, which cast shadows
_HEADLIGHT_DIFFUSE, _HEADLIGHT_AMBIENT = 0.25, 0.15
# defocus std in image pixels, darkening at the image corners, per-channel white balance, sensor noise std
_DEFOCUS, _VIGNETTE, _WHITE_BALANCE, _NOISE = 0.5, 0.12, (1.03, 1.0, 0.96), 1.5 / 255


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
    """Whether the ray tracer maps textures on `geom`: planes, and meshes with texture coordinates."""
    kind = m.geom_type[geom]
    return kind == mujoco.mjtGeom.mjGEOM_PLANE or (
        kind == mujoco.mjtGeom.mjGEOM_MESH and m.mesh_texcoordadr[m.geom_dataid[geom]] >= 0
    )


def _lit(m):
    """Copy of `m` whose lights cast shadows under a dim headlight."""
    m = deepcopy(m)
    m.light_castshadow[:] = True
    m.vis.headlight.diffuse[:] = _HEADLIGHT_DIFFUSE
    m.vis.headlight.ambient[:] = _HEADLIGHT_AMBIENT
    return m


def _blur(x, sigma):
    """`x` (h, w, c) convolved with a separable Gaussian of std `sigma` pixels, edges replicated."""
    r = jnp.arange(-2, 3)
    k = jnp.exp(-0.5 * (r / sigma) ** 2)
    k = k / k.sum()
    rows = jnp.pad(x, ((2, 2), (0, 0), (0, 0)), mode="edge")
    x = sum(k[i] * rows[i:i + x.shape[0]] for i in range(5))
    cols = jnp.pad(x, ((0, 0), (2, 2), (0, 0)), mode="edge")
    return sum(k[i] * cols[:, i:i + x.shape[1]] for i in range(5))


def _camera(rgb, key):
    """A camera's defocus, vignetting, white balance and sensor noise on an RGB image in [0, 1]."""
    h, w, _ = rgb.shape
    y, x = jnp.meshgrid(jnp.linspace(-1., 1., h), jnp.linspace(-1., 1., w), indexing="ij")
    rgb = _blur(rgb, _DEFOCUS) * (1. - _VIGNETTE * (x ** 2 + y ** 2) / 2.)[..., None] * jnp.asarray(_WHITE_BALANCE)
    return jnp.clip(rgb + _NOISE * jax.random.normal(key, rgb.shape), 0., 1.)


def _frame_key(d):
    """A key that differs between worlds and steps: of the state's time and joint positions."""
    bits = jax.lax.bitcast_convert_type(jnp.concatenate([jnp.ravel(d.time), d.qpos]).astype(jnp.float32), jnp.uint32)
    weights = jnp.arange(1, bits.size + 1, dtype=jnp.uint32) * jnp.uint32(2654435761)
    return jax.random.fold_in(jax.random.key(0), jnp.sum(bits * weights, dtype=jnp.uint32))


def _downsample(x, f):
    """`x` (f h, f w, c) averaged over f x f blocks."""
    h, w, c = x.shape
    return x.reshape(h // f, f, w // f, f, c).mean((1, 3))


def _geom_alpha(m):
    """Opacity of each geom: its material's alpha, or its own without a material."""
    return np.where(m.geom_matid >= 0, m.mat_rgba[m.geom_matid, 3], m.geom_rgba[:, 3])


def _textured(m):
    """
    Copy of `m` as the ray tracer shades it in one pass: sRGB textures in linear color, and the geoms whose textures
    it does not map in their materials' colors, untextured.
    """
    m = deepcopy(m)
    _linearize_srgb_textures(m)
    m.mat_shininess[:] = np.maximum(m.mat_shininess, _MIN_SHININESS)
    for g in _hit_textured(m):
        m.geom_rgba[g] = m.mat_rgba[m.geom_matid[g]]
        m.geom_matid[g] = -1
    return m


def _hit_textured(m):
    """Visible geoms with a texture the ray tracer does not map."""
    return [
        g for g in range(m.ngeom)
        if m.geom_matid[g] >= 0 and m.mat_texid[m.geom_matid[g], mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0
        and np.isin(m.geom_group[g], _VISIBLE_GEOM_GROUPS) and not _texture_is_mapped(m, g)
    ]


# texels along each side of the textures sampled at hit points
_TEXELS = 256


def _texture_image(m, tex):
    """Texture `tex` in linear color as a (_TEXELS, _TEXELS, 3) image in [0, 1]: a 2D texture, or a cube's first face."""
    w, h = m.tex_width[tex], m.tex_height[tex]
    img = _texels(m, tex)[:, :3].reshape(h, w, 3)
    if m.tex_type[tex] != mujoco.mjtTexture.mjTEXTURE_2D:
        img = img[:w]
    rows, cols = np.arange(_TEXELS) * img.shape[0] // _TEXELS, np.arange(_TEXELS) * img.shape[1] // _TEXELS
    return img[rows][:, cols] / 255.


class _HitTextures:
    """
    Textures of the geoms the ray tracer does not texture, sampled at each pixel's hit point in its geom's frame:
    projected onto the face of the geom's bounding box nearest the point, repeating as its material says, per meter
    with `texuniform` and per face otherwise.
    """
    def __init__(self, m, camera_names):
        lin = deepcopy(m)
        _linearize_srgb_textures(lin)
        geoms = _hit_textured(m)
        mats = m.geom_matid[geoms]
        textures = sorted({int(m.mat_texid[k, mujoco.mjtTextureRole.mjTEXROLE_RGB]) for k in mats})
        slot = -np.ones(m.ngeom, np.int32)
        slot[geoms] = [textures.index(int(m.mat_texid[k, mujoco.mjtTextureRole.mjTEXROLE_RGB])) for k in mats]
        self.slot = jnp.asarray(slot)
        self.atlas = jnp.asarray(np.stack([_texture_image(lin, t) for t in textures]) if textures else np.zeros((1, 1, 1, 3)),
                                 jnp.float32)
        mat = np.maximum(m.geom_matid, 0)
        self.center = jnp.asarray(m.geom_aabb[:, :3], jnp.float32)
        self.half = jnp.asarray(np.maximum(m.geom_aabb[:, 3:], 1e-6), jnp.float32)
        self.repeat = jnp.asarray(m.mat_texrepeat[mat], jnp.float32)
        self.uniform = jnp.asarray(m.mat_texuniform[mat])
        self.cams = jnp.asarray([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in camera_names])
        self.tan_half = jnp.asarray(np.tan(np.radians(m.cam_fovy[np.asarray(self.cams)]) / 2.), jnp.float32)
        self.any = bool(textures)

    def local_points(self, d, camera, geom, depth):
        """Each pixel's hit point in the frame of its geom `geom`, from its planar `depth` seen by camera index `camera`."""
        h, w = depth.shape
        cam = self.cams[camera]
        y, x = jnp.meshgrid(1. - (2. * jnp.arange(h) + 1.) / h, (2. * jnp.arange(w) + 1.) / w - 1., indexing="ij")
        t = self.tan_half[camera]
        ray = jnp.stack([x * t * w / h, y * t, -jnp.ones_like(x)], -1) * depth[..., None]
        world = d.cam_xpos[cam] + ray @ d.cam_xmat[cam].reshape(3, 3).T
        g = jnp.maximum(geom, 0)
        rot = d.geom_xmat[g].reshape(h, w, 3, 3)
        return jnp.einsum("hwji,hwj->hwi", rot, world - d.geom_xpos[g]) - self.center[g]

    def __call__(self, d, camera, geom, depth):
        """Each pixel's texture color, or 1 where its geom's texture is the ray tracer's or it has none."""
        g = jnp.maximum(geom, 0)
        slot = jnp.where(geom >= 0, self.slot[g], -1)
        p = self.local_points(d, camera, geom, depth)
        half = self.half[g]
        face = jnp.argmax(jnp.abs(p) / half, -1)
        u_axis, v_axis = jnp.where(face == 0, 1, 0), jnp.where(face == 2, 1, 2)
        pick = lambda a, i: jnp.take_along_axis(a, i[..., None], -1)[..., 0]  # noqa: E731
        span = lambda i: jnp.where(self.uniform[g], pick(p, i), (pick(p, i) / pick(half, i) + 1.) / 2.)  # noqa: E731
        rep = self.repeat[g]
        u, v = span(u_axis) * rep[..., 0], span(v_axis) * rep[..., 1]
        color = _bilinear(self.atlas, jnp.maximum(slot, 0), u, v)
        return jnp.where((slot >= 0)[..., None], color, 1.)


def _bilinear(atlas, slot, u, v):
    """Texels of images `atlas[slot]` at texture coordinates `u`, `v`, interpolated bilinearly, wrapping around."""
    n = atlas.shape[1]
    x, y = u * n - 0.5, (1. - v) * n - 0.5
    x0, y0 = jnp.floor(x), jnp.floor(y)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    at = lambda yy, xx: atlas[slot, yy.astype(jnp.int32) % n, xx.astype(jnp.int32) % n]  # noqa: E731
    top = at(y0, x0) * (1 - fx) + at(y0, x0 + 1) * fx
    bottom = at(y0 + 1, x0) * (1 - fx) + at(y0 + 1, x0 + 1) * fx
    return top * (1 - fy) + bottom * fy


class _Shading:
    """The color, depth, and frontmost geom of each pixel of one scene, lit and textured in one ray-traced pass."""
    def __init__(self, m, context, graph_mode):
        m = _textured(m)
        self.model = mjx.put_model(m, impl="warp", graph_mode=graph_mode)
        n = len(context["cam_active"])
        self.context = mjx.create_render_context(
            m, render_depth=[True] * n, render_seg=True, use_textures=True, **context
        )

    def __call__(self, d):
        """Color, depth, and segmentation buffers of the world `d`."""
        rc = self.context.pytree()
        color, depth, seg, _ = mjx.render_with_segmentation(self.model, mjx.refit_bvh(self.model, d, rc), rc)
        # an unbatched call returns the buffers of every world, the first of which is `d`'s
        if color.ndim == 2:
            color, depth, seg = color[0], depth[0] if depth.ndim == 2 else depth, seg[0]
        return color, depth, seg

    def rgb(self, camera, color):
        """RGB image in [0, 1] of camera index `camera` from the color buffer."""
        return mjx.get_rgb(self.context.pytree(), camera, color)


def _without_markers(m):
    """Copy of `m` without robosuite's end-effector target markers."""
    m = deepcopy(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    m.geom_group[[g for g, n in enumerate(names) if "eef_target" in n]] = _HIDDEN_GEOM_GROUP
    return m


class CameraRenderer:
    """
    RGB and depth images of named cameras, shaded as the module describes. Translucent geoms are blended over the
    scene behind them with their alpha, one layer deep, and occlude like opaque geoms in depth.
    """
    def __init__(
        self, m, camera_names: Sequence[str], height: int, width: int, depth: bool, nworld: int, graph_mode
    ):
        """Calls are unbatched or mapped over exactly `nworld` worlds."""
        self._camera_names = list(camera_names)
        self._height, self._width, self._depth = height, width, depth
        self._far = float(m.vis.map.zfar * m.stat.extent)
        m = _without_markers(_lit(m))
        self._textures = _HitTextures(m, self._camera_names)
        n = len(self._camera_names)
        context = dict(
            nworld=nworld, cam_res=[(SUPERSAMPLE * width, SUPERSAMPLE * height)] * n, cam_active=self._camera_names,
            render_rgb=True, use_shadows=SHADOWS, enabled_geom_groups=_VISIBLE_GEOM_GROUPS,
        )
        alpha = _geom_alpha(m)
        translucent = (alpha < 1.) & np.isin(m.geom_group, _VISIBLE_GEOM_GROUPS)
        self._scene = _Shading(m, context, graph_mode)
        self._behind = self._alpha = None
        if translucent.any():
            behind = deepcopy(m)
            behind.geom_group[translucent] = _HIDDEN_GEOM_GROUP
            self._behind = _Shading(behind, context, graph_mode)
            self._alpha = jnp.asarray(np.where(translucent, alpha, 1.), jnp.float32)

    def __call__(self, d):
        """
        Images of the world `d`: uint8 RGB images of shape (height, width, 3) under "<camera>_image" and planar
        depth in meters of shape (height, width, 1) under "<camera>_depth", both upright.
        """
        key = _frame_key(d)
        color, depth, seg = self._scene(d)
        behind = self._behind(d)[0] if self._behind is not None else None
        rc = self._scene.context.pytree()
        obs = {}
        for i, name in enumerate(self._camera_names):
            rgb = self._scene.rgb(i, color)
            front = mjx.get_segmentation(rc, i, seg)
            z = mjx.get_depth(rc, i, depth, self._far) * self._far
            if self._textures.any:
                rgb = rgb * self._textures(d, i, front, z[..., 0])
            if behind is not None:
                alpha = jnp.where(front >= 0, self._alpha[jnp.maximum(front, 0)], 1.)[..., None]
                rgb = alpha * rgb + (1. - alpha) * self._behind.rgb(i, behind)
            rgb = _camera(_downsample(rgb, SUPERSAMPLE), jax.random.fold_in(key, i))
            obs[f"{name}_image"] = jnp.round(rgb * 255.).astype(jnp.uint8)
            if self._depth:
                z = z[::SUPERSAMPLE, ::SUPERSAMPLE]
                obs[f"{name}_depth"] = jnp.where(z > 0., z, self._far)
        return obs
