from __future__ import annotations

import numpy as np
import pytest

from src.data.toy_physics import ToyPhysicsDataset


def test_toy_dataset_first_loss_target_is_k_plus_one_frame() -> None:
    context_length = 4
    dataset = ToyPhysicsDataset(
        num_episodes=1,
        episode_length=8,
        context_length=context_length,
        image_size=16,
    )

    sample = dataset[0]

    assert sample["context"].shape[0] == context_length
    assert sample["context_start"] == 0
    assert sample["context_end"] == context_length - 1
    assert sample["target_time"] == context_length
    assert sample["target_frame_number"] == context_length + 1
    assert dataset.first_target_index == context_length
    assert dataset.first_target_frame_number == context_length + 1


def test_toy_dataset_never_creates_pre_context_loss_samples() -> None:
    context_length = 3
    dataset = ToyPhysicsDataset(
        num_episodes=2,
        episode_length=10,
        context_length=context_length,
        image_size=16,
    )

    for _episode_idx, context_end in dataset.index:
        target_time = context_end + 1
        assert target_time >= context_length


def test_phyworld_dataset_window_metadata_uses_k_plus_one_target(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")

    context_length = 3
    path = tmp_path / "minimal_phyworld.hdf5"
    variable_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "w") as file:
        video_group = file.create_group("video_streams")
        position_group = file.create_group("position_streams")
        init_group = file.create_group("init_streams")
        video_group.create_dataset("00000", shape=(2,), dtype=variable_uint8)
        position_group.create_dataset("00000", data=np.zeros((2, 32, 2), dtype=np.float32))
        init_group.create_dataset("00000", data=np.zeros((2, 2), dtype=np.float32))

    from src.data.phyworld import PhyWorldUniformMotionDataset

    dataset = PhyWorldUniformMotionDataset(
        hdf5_path=path,
        context_length=context_length,
        image_size=16,
    )

    assert dataset.first_target_index == context_length
    assert dataset.first_target_frame_number == context_length + 1
    assert dataset._samples[0] == (0, context_length - 1)
    assert all(context_end + 1 >= context_length for _video_idx, context_end in dataset._samples)
    dataset.close()
