from dataclasses import dataclass, field
from math import ceil


@dataclass(slots=True)
class GridConfig:
    width: int
    height: int
    cols: int
    rows: int
    cell_w: int = field(init=False)
    cell_h: int = field(init=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            msg = "width and height must be > 0"
            raise ValueError(msg)
        if self.cols <= 0 or self.rows <= 0:
            msg = "cols and rows must be > 0"
            raise ValueError(msg)

        self.cell_w = ceil(self.width / self.cols)
        self.cell_h = ceil(self.height / self.rows)

    @property
    def n_cells(self) -> int:
        return self.cols * self.rows

    @property
    def cell_area(self) -> int:
        return self.cell_w * self.cell_h

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def cell_index(self, row: int, col: int) -> int:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            msg = f"cell out of bounds (row={row}, col={col})"
            raise ValueError(msg)
        return row * self.cols + col

    def cell_coords(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.n_cells:
            msg = f"cell index out of bounds (index={index})"
            raise ValueError(msg)
        return divmod(index, self.cols)

    def cell_index_for_xy(self, x: int, y: int) -> int:
        if x < 0:
            x = 0
        elif x >= self.width:
            x = self.width - 1

        if y < 0:
            y = 0
        elif y >= self.height:
            y = self.height - 1

        col = min(x // self.cell_w, self.cols - 1)
        row = min(y // self.cell_h, self.rows - 1)
        return row * self.cols + col

    def cell_bounds(self, row: int, col: int) -> tuple[int, int, int, int]:
        idx = self.cell_index(row, col)
        row, col = self.cell_coords(idx)
        x1 = col * self.cell_w
        y1 = row * self.cell_h
        x2 = min(x1 + self.cell_w, self.width)
        y2 = min(y1 + self.cell_h, self.height)
        return x1, y1, x2, y2


def grid_line_positions(cfg: GridConfig) -> tuple[list[int], list[int]]:
    """
    Return x and y pixel positions for grid boundaries (excluding outer border).
    """
    x_lines = [
        min((col + 1) * cfg.cell_w, cfg.width)
        for col in range(cfg.cols - 1)
    ]
    y_lines = [
        min((row + 1) * cfg.cell_h, cfg.height)
        for row in range(cfg.rows - 1)
    ]
    return x_lines, y_lines
