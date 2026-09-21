import cv2

from .config import CAMERA_SOURCES


class CameraSourceError(RuntimeError):
    pass


class CameraSourceSwitcher:
    """Thin wrapper around cv2.VideoCapture that can switch camera sources
    at runtime by releasing the current capture and opening the selected one.

    Switching is fail-safe: if the new source cannot be opened, the previous
    capture is left untouched and CameraSourceError is raised.
    """

    def __init__(
        self,
        sources: dict[str, int] | None = None,
        initial_source: str = "",
        initial_index: int | None = None,
    ) -> None:
        self._sources = list((sources if sources is not None else CAMERA_SOURCES).items())
        if not self._sources:
            raise CameraSourceError("No camera sources configured.")
        self._capture = None
        self._current_index = self._resolve_initial(initial_source, initial_index)
        self.open_current()

    @property
    def name(self) -> str:
        return self._sources[self._current_index][0]

    @property
    def index(self) -> int:
        return self._sources[self._current_index][1]

    @property
    def source_names(self) -> list[str]:
        return [name for name, _ in self._sources]

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def _resolve_initial(self, initial_source: str, initial_index: int | None) -> int:
        names = [name for name, _ in self._sources]
        if initial_source and initial_source in names:
            return names.index(initial_source)
        for position, (_, index) in enumerate(self._sources):
            if initial_index is not None and index == initial_index:
                return position
        return 0

    def open_current(self) -> None:
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise CameraSourceError(
                f"Unable to open camera source '{self.name}' at index {self.index}."
            )
        if self._capture is not None:
            self._capture.release()
        self._capture = capture

    def switch_next(self) -> str:
        next_index = (self._current_index + 1) % len(self._sources)
        previous_index = self._current_index
        self._current_index = next_index
        try:
            self.open_current()
        except CameraSourceError:
            self._current_index = previous_index
            raise
        return self.name

    def switch_to_name(self, name: str) -> str:
        names = {source_name: position for position, (source_name, _) in enumerate(self._sources)}
        if name not in names:
            raise CameraSourceError(f"Unknown camera source name '{name}'.")
        previous_index = self._current_index
        self._current_index = names[name]
        try:
            self.open_current()
        except CameraSourceError:
            self._current_index = previous_index
            raise
        return self.name

    def read(self):
        return self._capture.read()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None