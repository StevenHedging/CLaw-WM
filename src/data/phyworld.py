"""HDF5 adapter for the public PhyWorld uniform-motion dataset."""

from __future__ import annotations

import io
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PhyWorldUniformMotionDataset(Dataset[dict[str, torch.Tensor | int]]):
    """Frame-wise samples from PhyWorld's uniform-motion HDF5 files.

    Each HDF5 file stores MP4 bytes under ``video_streams`` and ground-truth
    positions under ``position_streams``. The returned state is normalized
    ``[x, y, vx, vy]`` with velocity estimated by finite differences.

    For context length ``K``, the first training target has zero-based index
    ``K``. Equivalently, the first supervised target is the one-based
    ``K+1``-th frame.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        context_length: int = 3,
        image_size: int = 64,
        max_videos: int | None = None,
        state_scale: float = 16.0,
        cache_size: int = 64,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Missing PhyWorld HDF5 file: {self.hdf5_path}")
        self.context_length = int(context_length)
        self.image_size = int(image_size)
        self.state_scale = float(state_scale)
        self.cache_size = int(cache_size)
        self._file: h5py.File | None = None
        self._video_cache: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()

        with h5py.File(self.hdf5_path, "r") as file:
            group_keys = sorted(file["video_streams"].keys())
            self._video_refs: list[tuple[str, int]] = []
            for group_key in group_keys:
                count = int(file["video_streams"][group_key].shape[0])
                for local_idx in range(count):
                    self._video_refs.append((group_key, local_idx))
                    if max_videos is not None and len(self._video_refs) >= int(max_videos):
                        break
                if max_videos is not None and len(self._video_refs) >= int(max_videos):
                    break
            first_group, first_idx = self._video_refs[0]
            num_frames = int(file["position_streams"][first_group][first_idx].shape[0])

        if num_frames <= self.context_length:
            raise ValueError("context_length must be smaller than the number of frames")
        self.num_frames = num_frames
        self.first_target_index = self.context_length
        self.first_target_frame_number = self.context_length + 1
        self._samples = [
            (video_idx, time)
            for video_idx in range(len(self._video_refs))
            for time in range(self.context_length - 1, self.num_frames - 1)
        ]

    def __len__(self) -> int:
        """Return the number of context/target windows."""

        return len(self._samples)

    def __getstate__(self) -> dict[str, Any]:
        """Avoid pickling an open HDF5 handle if DataLoader workers are used."""

        state = self.__dict__.copy()
        state["_file"] = None
        state["_video_cache"] = OrderedDict()
        return state

    @property
    def file(self) -> h5py.File:
        """Lazily open the HDF5 file."""

        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def close(self) -> None:
        """Close the lazy HDF5 handle."""

        if self._file is not None:
            self._file.close()
            self._file = None

    def _read_video(self, group_key: str, local_idx: int) -> torch.Tensor:
        cache_key = (group_key, local_idx)
        if cache_key in self._video_cache:
            frames = self._video_cache.pop(cache_key)
            self._video_cache[cache_key] = frames
            return frames

        video_bytes = self.file["video_streams"][group_key][local_idx].tobytes()
        frames_np = iio.imread(io.BytesIO(video_bytes), extension=".mp4")
        frames = torch.from_numpy(np.asarray(frames_np)).permute(0, 3, 1, 2).float() / 255.0
        if frames.shape[-1] != self.image_size or frames.shape[-2] != self.image_size:
            frames = F.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        self._video_cache[cache_key] = frames
        while len(self._video_cache) > self.cache_size:
            self._video_cache.popitem(last=False)
        return frames

    def _state_at(self, positions: np.ndarray, time: int) -> torch.Tensor:
        position = positions[time].astype(np.float32)
        prev_position = positions[max(time - 1, 0)].astype(np.float32)
        next_position = positions[min(time + 1, positions.shape[0] - 1)].astype(np.float32)
        velocity = next_position - position if time == 0 else position - prev_position
        state = np.concatenate([position, velocity], axis=0) / self.state_scale
        return torch.from_numpy(state.astype(np.float32))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int]:
        """Return one frame-wise PhyWorld sample."""

        video_idx, time = self._samples[item]
        group_key, local_idx = self._video_refs[video_idx]
        frames = self._read_video(group_key, local_idx)
        positions = np.asarray(self.file["position_streams"][group_key][local_idx])
        start = time - self.context_length + 1
        return {
            "context": frames[start : time + 1],
            "target": frames[time + 1],
            "state": self._state_at(positions, time),
            "next_state": self._state_at(positions, time + 1),
            "episode_id": video_idx,
            "time": time,
            "context_start": start,
            "context_end": time,
            "target_time": time + 1,
            "target_frame_number": time + 2,
            "dynamics_id": 0,
        }
