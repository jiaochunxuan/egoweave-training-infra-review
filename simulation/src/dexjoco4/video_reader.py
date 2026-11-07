from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


@dataclass(frozen=True)
class VideoSegment:
    path: Path
    frame_count: int
    start: int
    stop: int


def _set_quiet_av_logging() -> None:
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except Exception:
        return


def count_video_frames(path: Path | str) -> int:
    _set_quiet_av_logging()
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def _read_frame_sequential(path: Path, local_index: int) -> np.ndarray:
    _set_quiet_av_logging()
    import av

    if local_index < 0:
        raise IndexError(f"local_index must be non-negative, got {local_index}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame_number, frame in enumerate(container.decode(stream)):
            if frame_number == local_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"Frame {local_index} is out of range for {path}")


class SequentialVideoReader:
    """Read frames in-order without reopening and decoding a video per frame.

    DexJoCo windows are normally consumed in increasing frame order.  The old
    implementation cached each individual frame, but every cache miss decoded
    from frame zero, making a batch quadratic in its frame offset.  This reader
    keeps one decoder open and only reopens when access moves backwards.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        decoder_factory: Callable[[Path], Iterator[np.ndarray]] | None = None,
    ):
        self.path = Path(path)
        self._decoder_factory = decoder_factory or self._decode_video
        self._decoder: Iterator[np.ndarray] | None = None
        self._next_index = 0
        self._last_frame: np.ndarray | None = None
        self._last_index = -1

    @staticmethod
    def _decode_video(path: Path) -> Iterator[np.ndarray]:
        _set_quiet_av_logging()
        import av

        with av.open(str(path)) as container:
            yield from (frame.to_ndarray(format="rgb24") for frame in container.decode(container.streams.video[0]))

    def _reset(self) -> None:
        self._decoder = self._decoder_factory(self.path)
        self._next_index = 0
        self._last_frame = None
        self._last_index = -1

    def frame_at(self, local_index: int) -> np.ndarray:
        local_index = int(local_index)
        if local_index < 0:
            raise IndexError(f"local_index must be non-negative, got {local_index}")
        if self._decoder is None or local_index < self._next_index:
            self._reset()
        assert self._decoder is not None
        while self._next_index <= local_index:
            try:
                self._last_frame = np.asarray(next(self._decoder))
            except StopIteration as exc:
                raise IndexError(f"Frame {local_index} is out of range for {self.path}") from exc
            self._last_index = self._next_index
            self._next_index += 1
        assert self._last_frame is not None and self._last_index == local_index
        return self._last_frame.copy()


class SeekVideoReader:
    """Random-access reader that seeks to the nearest keyframe and decodes forward.

    Pure read-path acceleration: the returned frames are bit-identical to
    sequential decoding (verified by test_seek_reader.py).  Only used when the
    training data loader shuffles samples, where SequentialVideoReader would
    restart decoding from frame zero on nearly every access.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._container = None
        self._stream = None
        self._frame_pts: list[int] | None = None

    def _ensure_open(self) -> None:
        if self._container is not None:
            return
        _set_quiet_av_logging()
        import av

        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        # Build a frame_number -> pts map by demuxing packets only (no image
        # decoding), which is an order of magnitude faster than a full decode.
        pts_list: list[int] = []
        for packet in self._container.demux(self._stream):
            if packet.pts is not None:
                pts_list.append(packet.pts)
        self._frame_pts = pts_list
        # Rewind the demuxer so that subsequent seeks/decodes start clean.
        if pts_list:
            self._container.seek(pts_list[0], stream=self._stream, backward=True)

    def frame_at(self, local_index: int) -> np.ndarray:
        local_index = int(local_index)
        if local_index < 0:
            raise IndexError(f"local_index must be non-negative, got {local_index}")
        self._ensure_open()
        assert self._frame_pts is not None and self._container is not None and self._stream is not None
        if local_index >= len(self._frame_pts):
            raise IndexError(f"Frame {local_index} is out of range for {self.path}")
        target_pts = self._frame_pts[local_index]
        self._container.seek(target_pts, stream=self._stream, backward=True)
        for frame in self._container.decode(self._stream):
            if frame.pts == target_pts:
                return frame.to_ndarray(format="rgb24")
        raise IndexError(f"Frame {local_index} (pts={target_pts}) could not be decoded for {self.path}")

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
            self._stream = None


class ConcatenatedVideoReader:
    def __init__(
        self,
        paths: tuple[Path, ...] | list[Path],
        *,
        frame_counts: tuple[int, ...] | None = None,
        use_seek: bool = False,
    ):
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("ConcatenatedVideoReader requires at least one mp4 path")
        self._segments: tuple[VideoSegment, ...] | None = None
        self._readers: dict[Path, SequentialVideoReader] = {}
        self._seek_readers: dict[Path, SeekVideoReader] = {}
        self._use_seek = bool(use_seek)
        if frame_counts is not None:
            self._segments = self._make_segments(tuple(int(count) for count in frame_counts))

    def _make_segments(self, frame_counts: tuple[int, ...]) -> tuple[VideoSegment, ...]:
        if len(frame_counts) != len(self.paths):
            raise ValueError(f"Expected {len(self.paths)} frame counts, got {len(frame_counts)}")
        segments = []
        start = 0
        for path, frame_count in zip(self.paths, frame_counts, strict=True):
            stop = start + int(frame_count)
            segments.append(VideoSegment(path=path, frame_count=int(frame_count), start=start, stop=stop))
            start = stop
        return tuple(segments)

    def _ensure_segments(self) -> tuple[VideoSegment, ...]:
        if self._segments is None:
            self._segments = self._make_segments(tuple(count_video_frames(path) for path in self.paths))
        return self._segments

    def ensure_segments(self) -> None:
        """Public hook: precompute the segment map in the parent process."""
        self._ensure_segments()

    def _segment_for_index(self, global_index: int) -> tuple[Path, int]:
        if global_index < 0:
            raise IndexError(f"global_index must be non-negative, got {global_index}")
        if global_index == 0 and self._segments is None:
            return self.paths[0], 0
        segments = self._ensure_segments()
        stops = [segment.stop for segment in segments]
        segment_index = bisect_right(stops, global_index)
        if segment_index >= len(segments):
            raise IndexError(f"Frame {global_index} is out of range; total={segments[-1].stop}")
        segment = segments[segment_index]
        return segment.path, global_index - segment.start

    def frame_at(self, global_index: int) -> np.ndarray:
        path, local_index = self._segment_for_index(int(global_index))
        if self._use_seek:
            reader = self._seek_readers.setdefault(path, SeekVideoReader(path))
        else:
            reader = self._readers.setdefault(path, SequentialVideoReader(path))
        return reader.frame_at(int(local_index))

    def close(self) -> None:
        """Release any seek-based decoder containers held by this reader."""

        for reader in self._seek_readers.values():
            reader.close()
        self._seek_readers.clear()
        self._readers.clear()
