"""
Tests of the dataset tools and of the transferred demonstrations in datasets/warp against the released robomimic
datasets, which are looked up under $ROBOMIMIC_DATA/<task>/<type>/.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

from robomimic.data import hdf5, registry
from robomimic.data.observations import add_observations
from robomimic.env import TASK_OF_ENV_NAME, RobomimicPOMDP

PACKED = Path(__file__).parent.parent / "datasets" / "warp"
TRANSFERRED = ["lift_ph", "can_ph", "square_mh", "transport_ph", "tool_hang_ph"]


@pytest.fixture(scope="session")
def lift_test_dataset():
    folder = Path(__file__).parent / "assets"
    path = folder / "test_v15.hdf5"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        registry.download_file_from_hf(
            repo_id=registry.HF_REPO_ID, filename="test/test_v15.hdf5", download_dir=folder, check_overwrite=False
        )
    return path


def _released(name, filename="demo_v15.hdf5"):
    task, dataset_type = name.rsplit("_", 1)
    path = registry.DATA_DIR / task / dataset_type / filename
    if not path.exists():
        pytest.skip(f"no {path}")
    return path


def test_packed_demos_unpack_to_the_same_demos(lift_test_dataset, tmp_path):
    hdf5.pack_demos(lift_test_dataset, tmp_path / "demos.npz")
    hdf5.unpack_demos(tmp_path / "demos.npz", tmp_path / "demos.hdf5")
    with h5py.File(lift_test_dataset, "r") as original, h5py.File(tmp_path / "demos.hdf5", "r") as unpacked:
        assert set(unpacked["data"]) == set(original["data"])
        assert set(unpacked["mask"]) == set(original["mask"])
        assert unpacked["data"].attrs["env_args"] == original["data"].attrs["env_args"]
        for name in original["data"]:
            demo, recorded = unpacked[f"data/{name}"], original[f"data/{name}"]
            assert demo.attrs["model_file"] == recorded.attrs["model_file"]
            np.testing.assert_allclose(demo["states"][()], recorded["states"][()], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(demo["actions"][()], recorded["actions"][()].astype(np.float32))
            np.testing.assert_array_equal(demo["dones"][()], recorded["dones"][()])



@pytest.mark.parametrize("name", TRANSFERRED)
def test_transferred_demos_match_the_released_demos_and_all_succeed(name, tmp_path):
    hdf5.unpack_demos(PACKED / f"{name}.npz", tmp_path / "transferred.hdf5")
    with h5py.File(_released(name), "r") as released, h5py.File(tmp_path / "transferred.hdf5", "r") as transferred:
        assert set(transferred["data"]) == set(released["data"])
        assert set(transferred["mask"]) == set(released["mask"])
        for key in released["mask"]:
            assert sorted(transferred[f"mask/{key}"][()]) == sorted(released[f"mask/{key}"][()]), key
        for demo in released["data"]:
            assert transferred[f"data/{demo}/actions"].shape == released[f"data/{demo}/actions"].shape, demo
            assert transferred[f"data/{demo}"].attrs["success"], demo


def test_observations_of_transferred_demos_have_the_released_keys_and_shapes(tmp_path):
    dataset = tmp_path / "lift.hdf5"
    hdf5.unpack_demos(PACKED / "lift_ph.npz", dataset)
    with h5py.File(dataset, "a") as f:
        for demo in list(f["data"])[2:]:
            del f[f"data/{demo}"]
    add_observations(dataset)
    with h5py.File(_released("lift_ph", "low_dim_v15.hdf5"), "r") as released, h5py.File(dataset, "r") as f:
        for demo in f["data"]:
            for group in ("obs", "next_obs"):
                assert set(f[f"data/{demo}/{group}"]) == set(released[f"data/{demo}/{group}"])
                for key in f[f"data/{demo}/{group}"]:
                    assert f[f"data/{demo}/{group}/{key}"].shape == released[f"data/{demo}/{group}/{key}"].shape


def test_observations_of_states_are_those_of_the_pomdp(tmp_path):
    dataset = tmp_path / "lift.hdf5"
    hdf5.unpack_demos(PACKED / "lift_ph.npz", dataset)
    with h5py.File(dataset, "a") as f:
        for demo in list(f["data"])[1:]:
            del f[f"data/{demo}"]
    add_observations(dataset)
    with h5py.File(dataset, "r") as f:
        demo = f[f"data/{next(iter(f['data']))}"]
        env = RobomimicPOMDP("lift", model_xml=demo.attrs["model_file"], max_worlds=1)
        t = len(demo["states"]) // 2
        state = env.state_from_flat(demo["states"][t])._replace(cache=env.state_from_flat(demo["states"][t - 1]).cache)
        for key, value in env.observe(None, state, None).items():
            np.testing.assert_allclose(demo[f"obs/{key}"][t], np.asarray(value), atol=1e-6, err_msg=key)
            np.testing.assert_allclose(demo[f"next_obs/{key}"][t - 1], np.asarray(value), atol=1e-6, err_msg=key)
