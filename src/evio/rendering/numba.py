import numpy as np
from numba import njit


@njit(cache=True)
def fill_frame(frame_background: np.ndarray, base_intensity: int) -> None:
    height, width, _ = frame_background.shape
    for row in range(height):
        for col in range(width):
            frame_background[row, col, 0] = base_intensity
            frame_background[row, col, 1] = base_intensity
            frame_background[row, col, 2] = base_intensity


@njit(cache=True)
def paint_events(
    frame_bgr: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polarity_on: np.ndarray,
    on_blue: int,
    on_green: int,
    on_red: int,
    off_blue: int,
    off_green: int,
    off_red: int,
) -> None:
    height, width, _ = frame_bgr.shape
    event_count = x_coords.size
    for event_index in range(event_count):
        x_coord = int(x_coords[event_index])
        y_coord = int(y_coords[event_index])
        if 0 <= x_coord < width and 0 <= y_coord < height:
            if polarity_on[event_index]:
                frame_bgr[y_coord, x_coord, 0] = on_blue
                frame_bgr[y_coord, x_coord, 1] = on_green
                frame_bgr[y_coord, x_coord, 2] = on_red
            else:
                frame_bgr[y_coord, x_coord, 0] = off_blue
                frame_bgr[y_coord, x_coord, 1] = off_green
                frame_bgr[y_coord, x_coord, 2] = off_red


class NumbaEventRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        base_color: tuple[int, int, int] = (0, 0, 0),  # black — stationary
        on_color: tuple[int, int, int] = (0, 255, 0),  # green — brighter
        off_color: tuple[int, int, int] = (0, 0, 255),  # red — dimmer (BGR)
    ) -> None:
        self.width = width
        self.height = height
        self.frame_bgr = np.empty((height, width, 3), np.uint8)

        # assume grayscale base (like your original)
        self.base_intensity = int(base_color[0])
        self.on_color_bgr = (
            int(on_color[0]),
            int(on_color[1]),
            int(on_color[2]),
        )
        self.off_color_bgr = (
            int(off_color[0]),
            int(off_color[1]),
            int(off_color[2]),
        )

        # Trigger compilation up front with tiny arrays to avoid first-frame
        # stutter on the first frame.
        x_sample = np.zeros((1,), dtype=np.int32)
        y_sample = np.zeros((1,), dtype=np.int32)
        polarity_sample = np.zeros((1,), dtype=np.bool_)
        fill_frame(self.frame_bgr, self.base_intensity)
        paint_events(
            self.frame_bgr,
            x_sample,
            y_sample,
            polarity_sample,
            self.on_color_bgr[0],
            self.on_color_bgr[1],
            self.on_color_bgr[2],
            self.off_color_bgr[0],
            self.off_color_bgr[1],
            self.off_color_bgr[2],
        )

    def render(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        polarity_on: np.ndarray,
    ) -> np.ndarray:
        fill_frame(self.frame_bgr, self.base_intensity)
        paint_events(
            self.frame_bgr,
            x_coords.astype(np.int32, copy=False),
            y_coords.astype(np.int32, copy=False),
            polarity_on.astype(np.bool_, copy=False),
            self.on_color_bgr[0],
            self.on_color_bgr[1],
            self.on_color_bgr[2],
            self.off_color_bgr[0],
            self.off_color_bgr[1],
            self.off_color_bgr[2],
        )
        return self.frame_bgr
