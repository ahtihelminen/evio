from dataclasses import dataclass
from math import ceil


@dataclass(slots=True)
class GridCell:
    index: int
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(slots=True)
class Grid:
    width: int
    height: int
    cols: int
    rows: int
    cell_w: int
    cell_h: int
    cells: list[GridCell]

    def cell(self, *, index: int, row: int, col: int) -> GridCell:
        if index:
            return self.cells[index]
        return self.cells[row * self.cols + col]


def build_grid(width: int, height: int, cols: int, rows: int) -> Grid:
    if cols <= 0 or rows <= 0:
        msg = "cols and rows must be > 0"
        raise ValueError(msg)

    cell_w = ceil(width / cols)
    cell_h = ceil(height / rows)

    cells: list[GridCell] = []
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = min(x1 + cell_w, width)
            y2 = min(y1 + cell_h, height)
            idx = row * cols + col
            cells.append(
                GridCell(
                    index=idx,
                    row=row,
                    col=col,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return Grid(
        width=width,
        height=height,
        cols=cols,
        rows=rows,
        cell_w=cell_w,
        cell_h=cell_h,
        cells=cells,
    )


def grid_line_positions(grid: Grid) -> tuple[list[int], list[int]]:
    """
    Return x and y pixel positions for grid boundaries (excluding outer border).
    """
    x_lines = [
        cell.x2
        for cell in grid.cells
        if cell.row == 0 and cell.col < grid.cols - 1
    ]
    y_lines = [
        cell.y2
        for cell in grid.cells
        if cell.col == 0 and cell.row < grid.rows - 1
    ]
    return x_lines, y_lines
