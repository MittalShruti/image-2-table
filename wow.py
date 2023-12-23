# coding: utf-8
# coding: utf-8


# from img2table.document.base import Document
# from img2table.document.base.rotation import fix_rotation_image
# from img2table.tables.objects.extraction import ExtractedTable



# from img2table.document.base import Document
# from img2table.document.base.rotation import fix_rotation_image
# from img2table.ocr.pdf import PdfOCR


# coding: utf-8

# import xlsxwriter

# from img2table import Validations
# from img2table.tables.objects.extraction import ExtractedTable

# if typing.TYPE_CHECKING:
#     from img2table.ocr.base import OCRInstance
#     from img2table.tables.objects.table import Table


# coding: utf-8


dixon_q_test_confidence_dict = {
    0.9: {3: 0.941, 4: 0.765, 5: 0.642, 6: 0.56, 7: 0.507, 8: 0.468, 9: 0.437, 10: 0.412},
    0.95: {3: 0.970, 4: 0.829, 5: 0.71, 6: 0.625, 7: 0.568, 8: 0.526, 9: 0.493, 10: 0.466},
    0.99: {3: 0.994, 4: 0.926, 5: 0.821, 6: 0.74, 7: 0.68, 8: 0.634, 9: 0.598, 10: 0.568}
}



# from img2table.tables.metrics import compute_img_metrics
# from img2table.tables.objects.cell import Cell

# from img2table.tables.objects.extraction import TableCell, BBox


# from xlsxwriter.format import Format
# from xlsxwriter.worksheet import Worksheet
# from img2table.tables.objects.extraction import ExtractedTable, BBox

from dataclasses import dataclass
from functools import cached_property
import typing
import cv2
import polars as pl
import numpy as np
import io
from pathlib import Path
from typing import Tuple, Union, Dict, List, Optional, OrderedDict, NamedTuple, Any, Callable, Set
import math
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
import itertools
from functools import partial
import random
from queue import PriorityQueue

import fitz

from bs4 import BeautifulSoup

import pandas as pd
# from img2table.tables.processing.bordered_tables.tables.table_creation import cluster_to_table
# from img2table.tables.processing.bordered_tables.tables import cluster_to_table


class Validations:
    def __post_init__(self):
        """Run validation methods if declared.
        The validation method can be a simple check
        that raises ValueError or a transformation to
        the field value.
        The validation is performed by calling a function named:
            `validate_<field_name>(self, value, field) -> field.type`
        """
        for name, field in self.__dataclass_fields__.items():
            method = getattr(self, f"validate_{name}", None)
            setattr(self, name, method(getattr(self, name), field=field))


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int


class TableObject:
    def bbox(self, margin: int = 0, height_margin: int = 0, width_margin: int = 0) -> tuple:
        """
        Return bounding box corresponding to the object
        :param margin: general margin used for the bounding box
        :param height_margin: vertical margin used for the bounding box
        :param width_margin: horizontal margin used for the bounding box
        :return: tuple representing a bounding box
        """
        # Apply margin on bbox
        if margin != 0:
            bbox = (self.x1 - margin,
                    self.y1 - margin,
                    self.x2 + margin,
                    self.y2 + margin)
        else:
            bbox = (self.x1 - width_margin,
                    self.y1 - height_margin,
                    self.x2 + width_margin,
                    self.y2 + height_margin)


        return bbox

    @cached_property
    def height(self) -> int:
        return self.y2 - self.y1

    @cached_property
    def width(self) -> int:
        return self.x2 - self.x1

    @cached_property
    def area(self) -> int:
        return self.height * self.width

@dataclass
class TableCell:
    bbox: BBox
    value: Optional[str]

    def __hash__(self):
        return hash(repr(self))


class CellPosition(NamedTuple):
    cell: TableCell
    row: int
    col: int


@dataclass
class Cell(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    content: str = None

    @property
    def table_cell(self) -> TableCell:
        bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        return TableCell(bbox=bbox, value=self.content)

    def __hash__(self):
        return hash(repr(self))




@dataclass
class CellSpan:
    top_row: int
    bottom_row: int
    col_left: int
    col_right: int
    value: Optional[str]

    @property
    def colspan(self) -> int:
        return self.col_right - self.col_left + 1

    @property
    def rowspan(self) -> int:
        return self.bottom_row - self.top_row + 1

    @property
    def html_value(self) -> str:
        if self.value is not None:
            return self.value.replace("\n", "<br>")
        else:
            return ""

    @property
    def html(self) -> str:
        return f'<td colspan="{self.colspan}" rowspan="{self.rowspan}">{self.html_value}</td>'

    def html_cell_span(self) -> List["CellSpan"]:
        if self.colspan > 1 and self.rowspan > 1:
            # Check largest coordinate and split
            if self.colspan > self.rowspan:
                return [CellSpan(top_row=row_idx,
                                 bottom_row=row_idx,
                                 col_left=self.col_left,
                                 col_right=self.col_right,
                                 value=self.value)
                        for row_idx in range(self.top_row, self.bottom_row + 1)]
            else:
                return [CellSpan(top_row=self.top_row,
                                 bottom_row=self.bottom_row,
                                 col_left=col_idx,
                                 col_right=col_idx,
                                 value=self.value)
                        for col_idx in range(self.col_left, self.col_right + 1)]

        return [self]



@dataclass
class VertWS:
    x1: int
    x2: int
    whitespaces: List[Cell] = field(default_factory=lambda: [])
    positions: List[int] = field(default_factory=lambda: [])

    @property
    def y1(self):
        return min([ws.y1 for ws in self.whitespaces]) if self.whitespaces else 0

    @property
    def y2(self):
        return max([ws.y2 for ws in self.whitespaces]) if self.whitespaces else 0

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    @property
    def continuous(self):
        if self.positions:
            positions = sorted(self.positions)
            return all([p2 - p1 <= 1 for p1, p2 in zip(positions, positions[1:])])
        return False

    def add_ws(self, whitespaces: List[Cell]):
        self.whitespaces += whitespaces

    def add_position(self, position: int):
        self.positions.append(position)




@dataclass
class Rectangle:
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_cell(cls, cell: Cell) -> "Rectangle":
        return cls(x1=cell.x1, y1=cell.y1, x2=cell.x2, y2=cell.y2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def distance(self, other):
        return (self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2

    def overlaps(self, other):
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        return max(x_right - x_left, 0) * max(y_bottom - y_top, 0) > 0



class Row(TableObject):
    def __init__(self, cells: Union[Cell, List[Cell]]):
        if cells is None:
            raise ValueError("cells parameter is null")
        elif isinstance(cells, Cell):
            self._items = [cells]
        else:
            self._items = cells
        self._contours = []

    @property
    def items(self) -> List[Cell]:
        return self._items

    @property
    def nb_columns(self) -> int:
        return len(self.items)

    @property
    def x1(self) -> int:
        return min(map(lambda x: x.x1, self.items))

    @property
    def x2(self) -> int:
        return max(map(lambda x: x.x2, self.items))

    @property
    def y1(self) -> int:
        return min(map(lambda x: x.y1, self.items))

    @property
    def y2(self) -> int:
        return max(map(lambda x: x.y2, self.items))

    @property
    def v_consistent(self) -> bool:
        """
        Indicate if the row is vertically consistent (i.e all cells in row have the same vertical position)
        :return: boolean indicating if the row is vertically consistent
        """
        return all(map(lambda x: (x.y1 == self.y1) and (x.y2 == self.y2), self.items))

    def add_cells(self, cells: Union[Cell, List[Cell]]) -> "Row":
        """
        Add cells to existing row items
        :param cells: Cell object or list
        :return: Row object with cells added
        """
        if isinstance(cells, Cell):
            self._items += [cells]
        else:
            self._items += cells

        return self

    def split_in_rows(self, vertical_delimiters: List[int]) -> List["Row"]:
        """
        Split Row object into multiple objects based on vertical delimiters values
        :param vertical_delimiters: list of vertical delimiters values
        :return: list of splitted Row objects according to delimiters
        """
        # Create list of tuples for vertical boundaries
        row_delimiters = [self.y1] + vertical_delimiters + [self.y2]
        row_boundaries = [(i, j) for i, j in zip(row_delimiters, row_delimiters[1:])]

        # Create new list of rows
        l_new_rows = list()
        for boundary in row_boundaries:
            cells = list()
            for cell in self.items:
                _cell = copy.deepcopy(cell)
                _cell.y1, _cell.y2 = boundary
                cells.append(_cell)
            l_new_rows.append(Row(cells=cells))

        return l_new_rows

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            try:
                assert self.items == other.items
                return True
            except AssertionError:
                return False
        return False




class Table(TableObject):
    def __init__(self, rows: Union[Row, List[Row]], borderless: bool = False):
        if rows is None:
            self._items = []
        elif isinstance(rows, Row):
            self._items = [rows]
        else:
            self._items = rows
        self._title = None
        self._borderless = borderless

    @property
    def items(self) -> List[Row]:
        return self._items

    @property
    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title

    @property
    def nb_rows(self) -> int:
        return len(self.items)

    @property
    def nb_columns(self) -> int:
        return self.items[0].nb_columns if self.items else 0

    @property
    def x1(self) -> int:
        return min(map(lambda x: x.x1, self.items))

    @property
    def x2(self) -> int:
        return max(map(lambda x: x.x2, self.items))

    @property
    def y1(self) -> int:
        return min(map(lambda x: x.y1, self.items))

    @property
    def y2(self) -> int:
        return max(map(lambda x: x.y2, self.items))

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def remove_rows(self, row_ids: List[int]):
        """
        Remove rows by ids
        :param row_ids: list of row ids to be removed
        """
        # Get remaining rows
        remaining_rows = [idx for idx in range(self.nb_rows) if idx not in row_ids]

        if len(remaining_rows) > 1:
            # Check created gaps between rows
            gaps = [(id_row, id_next) for id_row, id_next in zip(remaining_rows, remaining_rows[1:])
                    if id_next - id_row > 1]

            for id_row, id_next in gaps:
                # Normalize y value between rows
                y_gap = int(round((self.items[id_row].y2 + self.items[id_next].y1) / 2))

                # Put y value in both rows
                for c in self.items[id_row].items:
                    setattr(c, "y2", max(c.y2, y_gap))
                for c in self.items[id_next].items:
                    setattr(c, "y1", min(c.y1, y_gap))

        # Remove rows
        for idx in reversed(row_ids):
            self.items.pop(idx)

    def remove_columns(self, col_ids: List[int]):
        """
        Remove columns by ids
        :param col_ids: list of column ids to be removed
        """
        # Get remaining cols
        remaining_cols = [idx for idx in range(self.nb_columns) if idx not in col_ids]

        if len(remaining_cols) > 1:
            # Check created gaps between columns
            gaps = [(id_col, id_next) for id_col, id_next in zip(remaining_cols, remaining_cols[1:])
                    if id_next - id_col > 1]

            for id_col, id_next in gaps:
                # Normalize x value between columns
                x_gap = int(round(np.mean([row.items[id_col].x2 + row.items[id_next].x1 for row in self.items]) / 2))

                # Put x value in both columns
                for row in self.items:
                    setattr(row.items[id_col], "x2", max(row.items[id_col].x2, x_gap))
                    setattr(row.items[id_next], "x1", min(row.items[id_next].x1, x_gap))

        # Remove columns
        for idx in reversed(col_ids):
            for id_row in range(self.nb_rows):
                self.items[id_row].items.pop(idx)

    def get_content(self, ocr_df: "OCRDataframe", min_confidence: int = 50) -> "Table":
        """
        Retrieve text from OCRDataframe object and reprocess table to remove empty rows / columns
        :param ocr_df: OCRDataframe object
        :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
        :return: Table object with data attribute containing dataframe
        """
        # Get content for each cell
        self = ocr_df.get_text_table(table=self, min_confidence=min_confidence)

        # Check for empty rows and remove if necessary
        empty_rows = list()
        for idx, row in enumerate(self.items):
            if all(map(lambda c: c.content is None, row.items)):
                empty_rows.append(idx)
        self.remove_rows(row_ids=empty_rows)

        # Check for empty columns and remove if necessary
        empty_cols = list()
        for idx in range(self.nb_columns):
            col_cells = [row.items[idx] for row in self.items]
            if all(map(lambda c: c.content is None, col_cells)):
                empty_cols.append(idx)
        self.remove_columns(col_ids=empty_cols)

        # Check for uniqueness of content
        unique_cells = set([cell for row in self.items for cell in row.items])
        if len(unique_cells) == 1:
            self._items = [Row(cells=self.items[0].items[0])]

        return self

    # @property
    # def extracted_table(self) -> ExtractedTable:
    #     bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
    #     content = OrderedDict({idx: [cell.table_cell for cell in row.items] for idx, row in enumerate(self.items)})
    #     return ExtractedTable(bbox=bbox, title=self.title, content=content)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            try:
                assert self.items == other.items
                if self.title is not None:
                    assert self.title == other.title
                else:
                    assert other.title is None
                return True
            except AssertionError:
                return False
        return False







@dataclass
class Line(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: Optional[int] = None

    @property
    def angle(self) -> float:
        delta_x = self.x2 - self.x1
        delta_y = self.y2 - self.y1

        return math.atan2(delta_y, delta_x) * 180 / np.pi

    @property
    def length(self) -> float:
        return np.sqrt(self.height ** 2 + self.width ** 2)

    @property
    def horizontal(self) -> bool:
        return self.angle % 180 == 0

    @property
    def vertical(self) -> bool:
        return self.angle % 180 == 90

    @property
    def dict(self):
        return {"x1": self.x1,
                "x2": self.x2,
                "y1": self.y1,
                "y2": self.y2,
                "width": self.width,
                "height": self.height,
                "thickness": self.thickness}

    @property
    def transpose(self) -> "Line":
        return Line(x1=self.y1, y1=self.x1, x2=self.y2, y2=self.x2, thickness=self.thickness)

    def reprocess(self):
        # Reallocate coordinates in proper order
        _x1 = min(self.x1, self.x2)
        _x2 = max(self.x1, self.x2)
        _y1 = min(self.y1, self.y2)
        _y2 = max(self.y1, self.y2)
        self.x1, self.x2, self.y1, self.y2 = _x1, _x2, _y1, _y2

        # Correct "almost" horizontal or vertical rows
        if abs(self.angle) <= 5:
            y_val = int(round((self.y1 + self.y2) / 2))
            self.y2 = self.y1 = y_val
        elif abs(self.angle - 90) <= 5:
            x_val = int(round((self.x1 + self.x2) / 2))
            self.x2 = self.x1 = x_val

        return self




@dataclass
class TableImage:
    img: np.ndarray
    min_confidence: int = 50
    char_length: float = None
    median_line_sep: float = None
    thresh: np.ndarray = None
    contours: List[Cell] = None
    lines: List[Line] = None
    tables: List[Table] = None

    def __post_init__(self):
        # Prepare image by removing eventual black background
        self.img = prepare_image(img=self.img)

        # Compute image metrics
        self.char_length, self.median_line_sep, self.contours = compute_img_metrics(img=self.img)

    @cached_property
    def white_img(self) -> np.ndarray:
        white_img = copy.deepcopy(self.img)

        # Draw white rows on detected rows
        for l in self.lines:
            if l.horizontal:
                cv2.rectangle(white_img, (l.x1 - l.thickness, l.y1), (l.x2 + l.thickness, l.y2), (255, 255, 255),
                              3 * l.thickness)
            elif l.vertical:
                cv2.rectangle(white_img, (l.x1, l.y1 - l.thickness), (l.x2, l.y2 + l.thickness), (255, 255, 255),
                              2 * l.thickness)

        return white_img

    def extract_bordered_tables(self, implicit_rows: bool = True):
        """
        Identify and extract bordered tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :return:
        """
        # Apply thresholding
        self.thresh = threshold_dark_areas(img=self.img, char_length=self.char_length)

        # Compute parameters for line detection
        minLinLength = maxLineGap = max(int(round(0.33 * self.median_line_sep)), 1) if self.median_line_sep else 10
        kernel_size = max(int(round(0.66 * self.median_line_sep)), 1) if self.median_line_sep else 20

        # Detect rows in image
        h_lines, v_lines = detect_lines(thresh=self.thresh,
                                        contours=self.contours,
                                        char_length=self.char_length,
                                        rho=0.3,
                                        theta=np.pi / 180,
                                        threshold=10,
                                        minLinLength=minLinLength,
                                        maxLineGap=maxLineGap,
                                        kernel_size=kernel_size)
        self.lines = h_lines + v_lines

        # Create cells from rows
        cells = get_cells(horizontal_lines=h_lines,
                          vertical_lines=v_lines)

        # Create tables from rows
        self.tables = get_tables(cells=cells,
                                 elements=self.contours,
                                 lines=self.lines,
                                 char_length=self.char_length)

        # If necessary, detect implicit rows
        if implicit_rows:
            self.tables = handle_implicit_rows(img=self.white_img,
                                               tables=self.tables,
                                               contours=self.contours)

        self.tables = [tb for tb in self.tables if tb.nb_rows * tb.nb_columns >= 2]

    def extract_borderless_tables(self):
        """
        Identify and extract borderless tables from image
        :return:
        """
        # Median line separation needs to be not null to extract borderless tables
        if self.median_line_sep is not None:
            # Extract borderless tables
            borderless_tbs = identify_borderless_tables(thresh=self.thresh,
                                                        char_length=self.char_length,
                                                        median_line_sep=self.median_line_sep,
                                                        lines=self.lines,
                                                        contours=self.contours,
                                                        existing_tables=self.tables)

            # Add to tables
            self.tables += [tb for tb in borderless_tbs if tb.nb_rows >= 2 and tb.nb_columns >= 3]

    def extract_tables(self, implicit_rows: bool = False, borderless_tables: bool = False) -> List[Table]:
        """
        Identify and extract tables from image
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :return: list of identified tables
        """
        # Extract bordered tables
        self.extract_bordered_tables(implicit_rows=implicit_rows)

        if borderless_tables:
            # Extract borderless tables
            self.extract_borderless_tables()

        return self.tables



@dataclass
class MockDocument:
    images: List[np.ndarray]


@dataclass
class Document(Validations):
    src: Union[str, Path, io.BytesIO, bytes]

    def validate_src(self, value, **_) -> Union[str, Path, io.BytesIO, bytes]:
        if not isinstance(value, (str, Path, io.BytesIO, bytes)):
            raise TypeError(f"Invalid type {type(value)} for src argument")
        return value

    def validate_detect_rotation(self, value, **_) -> int:
        if not isinstance(value, bool):
            raise TypeError(f"Invalid type {type(value)} for detect_rotation argument")
        return value

    def __post_init__(self):
        super(Document, self).__post_init__()
        # Initialize ocr_df
        self.ocr_df = None

        if not hasattr(self, "pages"):
            self.pages = None

        if isinstance(self.pages, list):
            self.pages = sorted(self.pages)

    @cached_property
    def bytes(self) -> bytes:
        if isinstance(self.src, bytes):
            return self.src
        elif isinstance(self.src, io.BytesIO):
            self.src.seek(0)
            return self.src.read()
        elif isinstance(self.src, str):
            with io.open(self.src, 'rb') as f:
                return f.read()

    @property
    def images(self) -> List[np.ndarray]:
        raise NotImplementedError

    # def get_table_content(self, tables: Dict[int, List["Table"]], ocr: "OCRInstance",
    #                       min_confidence: int) -> Dict[int, List[ExtractedTable]]:
    #     """
    #     Retrieve table content with OCR
    #     :param tables: dictionary containing extracted tables by page
    #     :param ocr: OCRInstance object used to extract table content
    #     :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
    #     :return: dictionary with page number as key and list of extracted tables as values
    #     """
    #     # Get pages where tables have been detected
    #     table_pages = [k for k, v in tables.items() if len(v) > 0]

    #     if (self.ocr_df is None and ocr is None) or len(table_pages) == 0:
    #         return {k: [tb.extracted_table for tb in v] for k, v in tables.items()}

    #     # Create document containing only pages
    #     ocr_doc = MockDocument(images=[self.images[page] for page in table_pages])

    #     # Get OCRDataFrame object
    #     if self.ocr_df is None and ocr is not None:
    #         self.ocr_df = ocr.of(document=ocr_doc)

    #     if self.ocr_df is None:
    #         return {k: [] for k in tables.keys()}

    #     # Retrieve table contents with ocr
    #     for idx, page in enumerate(table_pages):
    #         ocr_df_page = self.ocr_df.page(page_number=idx)
    #         # Get table content
    #         tables[page] = [table.get_content(ocr_df=ocr_df_page, min_confidence=min_confidence)
    #                         for table in tables[page]]

    #         # Filter relevant tables
    #         tables[page] = [table for table in tables[page] if max(table.nb_rows, table.nb_columns) >= 2]

    #         # Retrieve titles
    #         from img2table.tables.processing.text.titles import get_title_tables
    #         tables[page] = get_title_tables(img=self.images[page],
    #                                         tables=tables[page],
    #                                         ocr_df=ocr_df_page)

    #     # Reset OCR
    #     self.ocr_df = None

    #     return {k: [tb.extracted_table for tb in v
    #                 if (max(tb.nb_rows, tb.nb_columns) >= 2 and not tb._borderless)
    #                 or (tb.nb_rows >= 2 and tb.nb_columns >= 3)]
    #             for k, v in tables.items()}

    def extract_tables(self, implicit_rows: bool = False, borderless_tables: bool = False, min_confidence: int = 50) -> List[Table]:
    # def extract_tables(self, ocr: "OCRInstance" = None, implicit_rows: bool = False, borderless_tables: bool = False, min_confidence: int = 50) -> Dict[int, List[ExtractedTable]]:
        """
        Extract tables from document
        :param ocr: OCRInstance object used to extract table content
        :param implicit_rows: boolean indicating if implicit rows are splitted
        :param borderless_tables: boolean indicating if borderless tables should be detected
        :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
        :return: dictionary with page number as key and list of extracted tables as values
        """
        # Extract tables from document
        # from img2table.tables.image import TableImage
        tables = {idx: TableImage(img=img,
                                  min_confidence=min_confidence).extract_tables(implicit_rows=implicit_rows, borderless_tables=borderless_tables)
                  for idx, img in enumerate(self.images)}

        # Update table content with OCR if possible
        # tables = self.get_table_content(tables=tables,
        #                                 ocr=ocr,
        #                                 min_confidence=min_confidence)

        # # If pages have been defined, modify tables keys
        # if self.pages:
        #     tables = {self.pages[k]: v for k, v in tables.items()}

        return tables

    # def to_xlsx(self, dest: Union[str, Path, io.BytesIO], ocr: "OCRInstance" = None, implicit_rows: bool = False,
    #             borderless_tables: bool = False, min_confidence: int = 50) -> Optional[io.BytesIO]:
    #     """
    #     Create xlsx file containing all extracted tables from document
    #     :param dest: destination for xlsx file
    #     :param ocr: OCRInstance object used to extract table content
    #     :param implicit_rows: boolean indicating if implicit rows are splitted
    #     :param borderless_tables: boolean indicating if borderless tables should be detected
    #     :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
    #     :return: if a buffer is passed as dest arg, it is returned containing xlsx data
    #     """
    #     # Extract tables
    #     extracted_tables = self.extract_tables(ocr=ocr,
    #                                            implicit_rows=implicit_rows,
    #                                            borderless_tables=borderless_tables,
    #                                            min_confidence=min_confidence)
    #     extracted_tables = {0: extracted_tables} if isinstance(extracted_tables, list) else extracted_tables

    #     # Create workbook
    #     workbook = xlsxwriter.Workbook(dest, {'in_memory': True})

    #     # Create generic cell format
    #     cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
    #     cell_format.set_border()

    #     # For each extracted table, create a corresponding worksheet and populate it
    #     for page, tables in extracted_tables.items():
    #         for idx, table in enumerate(tables):
    #             # Create worksheet
    #             sheet = workbook.add_worksheet(name=f"Page {page + 1} - Table {idx + 1}")

    #             # Populate worksheet
    #             table._to_worksheet(sheet=sheet, cell_fmt=cell_format)

    #     # Close workbook
    #     workbook.close()

    #     # If destination is a BytesIO object, return it
    #     if isinstance(dest, io.BytesIO):
    #         dest.seek(0)
    #         return dest


@dataclass
class Image(Document):
    detect_rotation: bool = False

    def __post_init__(self):
        self.pages = None

        super(Image, self).__post_init__()

    @cached_property
    def images(self) -> List[np.ndarray]:
        img = cv2.imdecode(np.frombuffer(self.bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if self.detect_rotation:
            rotated_img, _ = fix_rotation_image(img=img)
            return [rotated_img]
        else:
            return [img]

    # def extract_tables(self, ocr: "OCRInstance" = None, implicit_rows: bool = False, borderless_tables: bool = False,
    #                    min_confidence: int = 50) -> List[ExtractedTable]:
    #     """
    #     Extract tables from document
    #     :param ocr: OCRInstance object used to extract table content
    #     :param implicit_rows: boolean indicating if implicit rows are splitted
    #     :param borderless_tables: boolean indicating if borderless tables should be detected
    #     :param min_confidence: minimum confidence level from OCR in order to process text, from 0 (worst) to 99 (best)
    #     :return: list of extracted tables
    #     """
    #     extracted_tables = super(Image, self).extract_tables(ocr=ocr,
    #                                                          implicit_rows=implicit_rows,
    #                                                          borderless_tables=borderless_tables,
    #                                                          min_confidence=min_confidence)
    #     return extracted_tables.get(0)




@dataclass
class PDF(Document):
    pages: List[int] = None
    detect_rotation: bool = False
    pdf_text_extraction: bool = True
    _rotated: bool = False
    _images: List[np.ndarray] = None

    def validate_pages(self, value, **_) -> Optional[List[int]]:
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Invalid type {type(value)} for pages argument")
            if not all(isinstance(x, int) for x in value):
                raise TypeError("All values in pages argument should be integers")
        return value

    def validate_pdf_text_extraction(self, value, **_) -> int:
        if not isinstance(value, bool):
            raise TypeError(f"Invalid type {type(value)} for pdf_text_extraction argument")
        return value

    def validate__rotated(self, value, **_) -> int:
        return value

    def validate__images(self, value, **_) -> int:
        return value

    @property
    def images(self) -> List[np.ndarray]:
        if self._images is not None:
            return self._images

        mat = fitz.Matrix(200 / 72, 200 / 72)
        doc = fitz.Document(stream=self.bytes, filetype='pdf')

        # Get all images
        images = list()
        for page_number in self.pages or range(doc.page_count):
            page = doc.load_page(page_id=page_number)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            # To grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Handle rotation if needed
            if self.detect_rotation:
                final, self._rotated = fix_rotation_image(img=gray)
            else:
                final, self._rotated = gray, False
            images.append(final)

        self._images = images
        return images

    # def get_table_content(self, tables: Dict[int, List["Table"]], ocr: "OCRInstance",
    #                       min_confidence: int) -> Dict[int, List["ExtractedTable"]]:
    #     if not self._rotated and self.pdf_text_extraction:
    #         # Get pages where tables have been detected
    #         table_pages = [self.pages[k] if self.pages else k for k, v in tables.items() if len(v) > 0]
    #         images = [self.images[k] for k, v in tables.items() if len(v) > 0]

    #         if table_pages:
    #             # Create PDF object for OCR
    #             pdf_ocr = PDF(src=self.bytes,
    #                           pages=table_pages,
    #                           _images=images,
    #                           _rotated=self._rotated)

    #             # Try to get OCRDataframe from PDF
    #             self.ocr_df = PdfOCR().of(document=pdf_ocr)

    #     return super(PDF, self).get_table_content(tables=tables, ocr=ocr, min_confidence=min_confidence)







@dataclass
class ImageSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    elements: List[Cell] = None
    whitespaces: List[Cell] = None
    position: int = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def set_elements(self, elements: List[Cell]):
        self.elements = elements

    def set_whitespaces(self, whitespaces: List[Cell]):
        self.whitespaces = whitespaces

    def __hash__(self):
        return hash(repr(self))


@dataclass
class TableSegment:
    table_areas: List[ImageSegment]

    @property
    def x1(self) -> int:
        return min([tb_area.x1 for tb_area in self.table_areas])

    @property
    def y1(self) -> int:
        return min([tb_area.y1 for tb_area in self.table_areas])

    @property
    def x2(self) -> int:
        return max([tb_area.x2 for tb_area in self.table_areas])

    @property
    def y2(self) -> int:
        return max([tb_area.y2 for tb_area in self.table_areas])

    @property
    def elements(self) -> List[Cell]:
        return [el for tb_area in self.table_areas for el in tb_area.elements]

    @property
    def whitespaces(self) -> List[Cell]:
        return [ws for tb_area in self.table_areas for ws in tb_area.whitespaces]


@dataclass
class DelimiterGroup:
    delimiters: List[Cell]
    elements: List[Cell] = field(default_factory=lambda: [])

    @property
    def x1(self) -> int:
        if self.delimiters:
            return min([d.x1 for d in self.delimiters])
        return 0

    @property
    def y1(self) -> int:
        if self.delimiters:
            return min([d.y1 for d in self.delimiters])
        return 0

    @property
    def x2(self) -> int:
        if self.delimiters:
            return max([d.x2 for d in self.delimiters])
        return 0

    @property
    def y2(self) -> int:
        if self.delimiters:
            return max([d.y2 for d in self.delimiters])
        return 0

    @property
    def bbox(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def add(self, delim: Cell):
        self.delimiters.append(delim)

    def __eq__(self, other):
        if isinstance(other, DelimiterGroup):
            try:
                assert set(self.delimiters) == set(other.delimiters)
                assert set(self.elements) == set(other.elements)
                return True
            except AssertionError:
                return False
        return False


@dataclass
class TableRow:
    cells: List[Cell]

    @property
    def x1(self) -> int:
        return min([c.x1 for c in self.cells])

    @property
    def y1(self) -> int:
        return min([c.y1 for c in self.cells])

    @property
    def x2(self) -> int:
        return max([c.x2 for c in self.cells])

    @property
    def y2(self) -> int:
        return max([c.y2 for c in self.cells])

    @property
    def v_center(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def overlaps(self, other: "TableRow") -> bool:
        # Compute y overlap
        y_top = max(self.y1, other.y1)
        y_bottom = min(self.y2, other.y2)

        return (y_bottom - y_top) / min(self.height, other.height) >= 0.33

    def merge(self, other: "TableRow") -> "TableRow":
        return TableRow(cells=self.cells + other.cells)

    def set_y_top(self, y_value: int):
        self.cells = [Cell(x1=c.x1, y1=y_value, x2=c.x2, y2=c.y2) for c in self.cells]

    def set_y_bottom(self, y_value: int):
        self.cells = [Cell(x1=c.x1, y1=c.y1, x2=c.x2, y2=y_value) for c in self.cells]

    def __eq__(self, other):
        if isinstance(other, TableRow):
            try:
                assert set(self.cells) == set(other.cells)
                return True
            except AssertionError:
                return False
        return False

    def __hash__(self):
        return hash(f"{self.x1},{self.y1},{self.x2},{self.y2}")







class Node:
    def __init__(self, key):
        self.key = key
        self.parent = self
        self.size = 1


class UnionFind(dict):
    def find(self, key):
        node = self.get(key, None)
        if node is None:
            node = self[key] = Node(key)
        else:
            while node.parent != node:
                # walk up & perform path compression
                node.parent, node = node.parent.parent, node.parent
        return node

    def union(self, key_a, key_b):
        node_a = self.find(key_a)
        node_b = self.find(key_b)
        if node_a != node_b:  # disjoint? -> join!
            if node_a.size < node_b.size:
                node_a.parent = node_b
                node_b.size += node_a.size
            else:
                node_b.parent = node_a
                node_a.size += node_b.size










def find_components(edges):
    forest = UnionFind()

    for edge in edges:
        edge = edge if len(edge) > 1 else list(edge) * 2
        forest.union(*edge)

    result = defaultdict(list)
    for key in forest.keys():
        root = forest.find(key)
        result[root.key].append(key)

    return list(result.values())


def compute_char_length(img: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Compute average character length based on connected components analysis
    :param img: image array
    :return: tuple with average character length and connected components array
    """
    # Thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    _, _, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Remove connected components with less than 5 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 5
    stats = stats[mask_pixels]

    if len(stats) == 0:
        return None, None

    # Create mask to remove connected components corresponding to the complete image
    mask_height = img.shape[0] > stats[:, cv2.CC_STAT_HEIGHT]
    mask_width = img.shape[1] > stats[:, cv2.CC_STAT_WIDTH]
    mask_img = mask_width & mask_height

    # Filter components based on aspect ratio
    mask_lower_ar = 0.5 < stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_ar = 2 > stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT]
    mask_ar = mask_lower_ar & mask_upper_ar

    stats = stats[mask_img & mask_ar]

    # Compute median width and height
    median_width = np.median(stats[:, cv2.CC_STAT_WIDTH])
    median_height = np.median(stats[:, cv2.CC_STAT_HEIGHT])

    # Compute bbox area bounds
    upper_bound = 4 * median_width * median_height
    lower_bound = 0.25 * median_width * median_height

    # Filter connected components according to their area
    mask_lower_area = lower_bound <= stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_area = upper_bound >= stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_area = mask_lower_area & mask_upper_area

    # Filter connected components from mask
    stats = stats[mask_area]

    if len(stats) > 0:
        # Compute average character length
        char_length = np.mean(stats[:, cv2.CC_STAT_WIDTH])

        return char_length, stats
    else:
        return None, None


def recompute_contours(cells_cc: List[Cell], df_contours: pl.LazyFrame) -> List[Cell]:
    """
    Recompute contours identified with original cells from connected components
    :param cells_cc: list of cells from connected components
    :param df_contours: dataframe containing contours
    :return: list of final contours
    """
    # Create dataframes for cells
    df_cells = pl.LazyFrame([{"x1_c": c.x1, "y1_c": c.y1, "x2_c": c.x2, "y2_c": c.y2}
                             for c in cells_cc])

    # Cross join and filters cells contained in contours
    df_contained_cells = (
        df_contours.join(df_cells, how="cross")
        .filter(pl.col("x1_c") >= pl.col("x1"),
                pl.col("y1_c") >= pl.col("y1"),
                pl.col("x2_c") <= pl.col("x2"),
                pl.col("y2_c") <= pl.col("y2"))
        .group_by("id")
        .agg(pl.min("x1_c").alias("x1"),
             pl.min("y1_c").alias("y1"),
             pl.max("x2_c").alias("x2"),
             pl.max("y2_c").alias("y2"))
    )

    # Create final contours
    final_contours = [Cell(x1=row.get('x1'), y1=row.get('y1'), x2=row.get('x2'), y2=row.get('y2'))
                      for row in df_contained_cells.collect().to_dicts()]

    return final_contours


def compute_median_line_sep(img: np.ndarray, cc: np.ndarray,
                            char_length: float) -> Tuple[Optional[float], Optional[List[Cell]]]:
    """
    Compute median separation between rows
    :param img: image array
    :param cc: connected components array
    :param char_length: average character length
    :return: median separation between rows
    """
    # Create image from connected components
    black_img = np.zeros(img.shape, np.uint8)
    cells_cc = [Cell(x1=c[cv2.CC_STAT_LEFT],
                     y1=c[cv2.CC_STAT_TOP],
                     x2=c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH],
                     y2=c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT]) for c in cc]
    for cell in cells_cc:
        cv2.rectangle(black_img, (cell.x1, cell.y1), (cell.x2, cell.y2), (255, 255, 255), -1)

    # Dilate image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(round(char_length)), 1), 1))
    dilate = cv2.dilate(black_img, kernel, iterations=1)

    # Find and map contours
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = list()
    for idx, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        contours.append({"id": idx, "x1": x, "y1": y, "x2": x + w, "y2": y + h})

    if len(contours) == 0:
        return None, []

    # Create contours dataframe
    df_contours = pl.LazyFrame(data=contours)

    # Cross join to get corresponding contours and filter on contours that corresponds horizontally
    df_h_cnts = (df_contours.join(df_contours, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right']) > 0)
                 )

    # Get contour which is directly below
    df_cnts_below = (df_h_cnts.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cum_sum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_cnts_below.collect().height == 0:
        return None, [Cell(x1=c.get('x1'), y1=c.get('y1'), x2=c.get('x2'), y2=c.get('y2')) for c in contours]

    # Compute median vertical distance between contours
    median_v_dist = (df_cnts_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                   - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect()
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    # Recompute contours
    final_contours = recompute_contours(cells_cc=cells_cc,
                                        df_contours=df_contours)

    return median_v_dist, final_contours


def compute_img_metrics(img: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[List[Cell]]]:
    """
    Compute metrics from image
    :param img: image array
    :return: average character length, median line separation and image contours
    """
    # Compute average character length based on connected components analysis
    char_length, cc_array = compute_char_length(img=img)

    if char_length is None:
        return None, None, None

    # Compute median separation between rows
    median_line_sep, contours = compute_median_line_sep(img=img, cc=cc_array, char_length=char_length)

    return char_length, median_line_sep, contours






def create_all_rectangles(cell_positions: List[CellPosition]) -> List[CellSpan]:
    """
    Create all possible rectangles from list of cell positions
    :param cell_positions: list of cell positions
    :return: list of CellSpan objects representing rectangle coordinates
    """
    # Get cell value
    cell_value = cell_positions[0].cell.value

    # Get bounding coordinates
    min_col = min(map(lambda x: x.col, cell_positions))
    max_col = max(map(lambda x: x.col, cell_positions))
    min_row = min(map(lambda x: x.row, cell_positions))
    max_row = max(map(lambda x: x.row, cell_positions))

    # Get largest rectangle fully covered by cell positions
    largest_area, area_coords, area_cell_pos = 0, None, None
    for col_left in range(min_col, max_col + 1):
        for col_right in range(col_left, max_col + 1):
            for top_row in range(min_row, max_row + 1):
                for bottom_row in range(top_row, max_row + 1):
                    # Get matching cell positions
                    matching_cell_pos = [cp for cp in cell_positions if col_left <= cp.col <= col_right
                                         and top_row <= cp.row <= bottom_row]

                    # Check if the rectangle is fully covered
                    fully_covered = len(matching_cell_pos) == (col_right - col_left + 1) * (bottom_row - top_row + 1)

                    # If rectangle is the largest, update values
                    if fully_covered and (len(matching_cell_pos) > largest_area):
                        largest_area = len(matching_cell_pos)
                        area_cell_pos = matching_cell_pos
                        cell_span = CellSpan(col_left=col_left,
                                             top_row=top_row,
                                             col_right=col_right,
                                             bottom_row=bottom_row,
                                             value=cell_value)

    # Get remaining cell positions
    remaining_cell_positions = [cp for cp in cell_positions if cp not in area_cell_pos]

    if remaining_cell_positions:
        # Get remaining rectangles
        return [cell_span] + create_all_rectangles(remaining_cell_positions)
    else:
        # Return coordinates
        return [cell_span]


# @dataclass
# class ExtractedTable:
#     bbox: BBox
#     title: Optional[str]
#     content: OrderedDict[int, List[TableCell]]

#     @property
#     def df(self) -> pd.DataFrame:
#         """
#         Create pandas DataFrame representation of the table
#         :return: pandas DataFrame containing table data
#         """
#         values = [[cell.value for cell in row] for k, row in self.content.items()]
#         return pd.DataFrame(values)

#     @property
#     def html(self) -> str:
#         """
#         Create HTML representation of the table
#         :return: HTML table
#         """
#         # Group cells based on hash (merged cells are duplicated over multiple rows/columns in content)
#         dict_cells = dict()
#         for id_row, row in self.content.items():
#             for id_col, cell in enumerate(row):
#                 cell_pos = CellPosition(cell=cell, row=id_row, col=id_col)
#                 dict_cells[hash(cell)] = dict_cells.get(hash(cell), []) + [cell_pos]

#         # Get list of cell spans
#         cell_span_list = [cell_span for _, cells in dict_cells.items()
#                           for cell_span in create_all_rectangles(cell_positions=cells)]
#         cell_span_list = [span for cell_span in cell_span_list for span in cell_span.html_cell_span()]

#         # Create HTML rows
#         rows_html = list()
#         for row_idx in range(len(self.content)):
#             # Get cells in row
#             row_cells = sorted([cell_span for cell_span in cell_span_list if cell_span.top_row == row_idx],
#                                key=lambda cs: cs.col_left)
#             html_row = "<tr>" + "".join([cs.html for cs in row_cells]) + "</tr>"
#             rows_html.append(html_row)

#         # Create HTML table
#         table_html = "<table>" + "".join(rows_html) + "</table>"

#         return BeautifulSoup(table_html).prettify().strip()

#     def _to_worksheet(self, sheet: Worksheet, cell_fmt: Optional[Format] = None):
#         """
#         Populate xlsx worksheet with table data
#         :param sheet: xlsxwriter Worksheet
#         :param cell_fmt: xlsxwriter cell format
#         """
#         # Group cells based on hash (merged cells are duplicated over multiple rows/columns in content)
#         dict_cells = dict()
#         for id_row, row in self.content.items():
#             for id_col, cell in enumerate(row):
#                 cell_pos = CellPosition(cell=cell, row=id_row, col=id_col)
#                 dict_cells[hash(cell)] = dict_cells.get(hash(cell), []) + [cell_pos]

#         # Write all cells to sheet
#         for c in dict_cells.values():
#             if len(c) == 1:
#                 cell_pos = c.pop()
#                 sheet.write(cell_pos.row, cell_pos.col, cell_pos.cell.value, cell_fmt)
#             else:
#                 # Get all rectangles
#                 for cell_span in create_all_rectangles(cell_positions=c):
#                     # Case of merged cells
#                     sheet.merge_range(first_row=cell_span.top_row,
#                                       first_col=cell_span.col_left,
#                                       last_row=cell_span.bottom_row,
#                                       last_col=cell_span.col_right,
#                                       data=cell_span.value,
#                                       cell_format=cell_fmt)

#         # Autofit worksheet
#         sheet.autofit()

#     def html_repr(self, title: Optional[str] = None) -> str:
#         """
#         Create HTML representation of the table
#         :param title: title of HTML paragraph
#         :return: HTML string
#         """
#         html = f"""{rf'<h3 style="text-align: center">{title}</h3>' if title else ''}
#                    <p style=\"text-align: center\">
#                        <b>Title:</b> {self.title or 'No title detected'}<br>
#                        <b>Bounding box:</b> x1={self.bbox.x1}, y1={self.bbox.y1}, x2={self.bbox.x2}, y2={self.bbox.y2}
#                    </p>
#                    <div align=\"center\">{self.df.to_html().replace("None", "")}</div>
#                    <hr>
#                 """
#         return html

#     def __repr__(self):
#         return f"ExtractedTable(title={self.title}, bbox=({self.bbox.x1}, {self.bbox.y1}, {self.bbox.x2}, " \
#                f"{self.bbox.y2}),shape=({len(self.content)}, {len(self.content[0])}))".strip()





def get_connected_components(img: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Identify connected components in image
    :param img: image array
    :return: list of connected components centroids and thresholded image
    """
    # Thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    _, _, stats, _ = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # Remove connected components with less than 5 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 5
    stats = stats[mask_pixels]

    # Compute median width and height
    median_width = np.median(stats[:, cv2.CC_STAT_WIDTH])
    median_height = np.median(stats[:, cv2.CC_STAT_HEIGHT])

    # Compute bbox area bounds
    upper_bound = 4 * median_width * median_height
    lower_bound = 0.25 * median_width * median_height

    # Filter connected components according to their area
    mask_lower_area = lower_bound < stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_area = upper_bound > stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_area = mask_lower_area & mask_upper_area

    # Filter components based on aspect ratio
    mask_lower_ar = 0.5 < stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_ar = 2 > stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT]
    mask_ar = mask_lower_ar & mask_upper_ar

    # Create general mask
    mask = mask_area & mask_ar

    # Get centroids from mask
    stats = stats[mask]
    centroids_x = stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] / 2
    centroids_y = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] / 2
    filtered_centroids = np.column_stack([centroids_x, centroids_y])

    return filtered_centroids, median_height, thresh


def get_relevant_angles(centroids: np.ndarray, ref_height: float, n_max: int = 5) -> List[float]:
    """
    Identify relevant angles from connected components centroids
    :param centroids: array of connected components centroids
    :param ref_height: reference height
    :param n_max: maximum number of returned angles
    :return: list of angle values
    """
    if len(centroids) == 0:
        return [0]

    # Create dataframe with centroids
    df_centroids = pl.LazyFrame(data=centroids, schema=['x1', 'y1'])

    # Cross join and keep only relevant pairs
    df_cross = (df_centroids.join(df_centroids, how='cross')
                .filter(pl.col('x1') != pl.col('x1_right'))
                .filter((pl.col('y1') - pl.col('y1_right')).abs() <= 10 * ref_height)
                )

    # Compute slopes and angles
    df_angles = (df_cross.with_columns(((pl.col('y1') - pl.col('y1_right')) / (pl.col('x1') - pl.col('x1_right'))
                                        ).round(3).alias('slope'))
                 .with_columns((pl.col('slope').arctan() * 180 / np.pi).alias('angle'))
                 .with_columns(pl.when(pl.col('angle').abs() <= 45)
                               .then(pl.col('angle'))
                               .otherwise(pl.min_horizontal(pl.col('angle') + 90, 90 - pl.col('angle')) * -pl.col('angle').sign())
                               .alias('angle')
                               )
                 )

    # Get n most represented angles
    most_likely_angles = (df_angles.group_by('angle')
                          .count()
                          .sort(by=['count', pl.col('angle').abs()], descending=[True, False])
                          .limit(n_max)
                          .collect(streaming=True)
                          .to_dicts()
                          )

    if most_likely_angles:
        if most_likely_angles[0].get('angle') == 0:
            return [0]
        else:
            return sorted(list(set([angle.get('angle') for angle in most_likely_angles
                                    if angle.get('count') >= 0.25 * max([a.get('count') for a in most_likely_angles])])))
    return [0]


def angle_dixon_q_test(angles: List[float], confidence: float = 0.9) -> float:
    """
    Compute best angle according to Dixon Q test
    :param angles: list of possible angles
    :param confidence: confidence level for outliers (0.9, 0.95, 0.99)
    :return: estimated angle
    """
    # Get dict of Q crit corresponding to confidence level
    dict_q_crit = dixon_q_test_confidence_dict.get(confidence)

    while len(angles) >= 3:
        # Compute range
        rng = angles[-1] - angles[0]

        # Get outlier and compute diff with closest angle
        diffs = [abs(nexxt - prev) for prev, nexxt in zip(angles, angles[1:])]
        idx_outlier = 0 if np.argmax(diffs) == 0 else len(angles) - 1
        gap = np.max(diffs)

        # Compute Qexp and compare to Qcrit
        q_exp = gap / rng

        if q_exp > dict_q_crit.get(len(angles)):
            angles.pop(idx_outlier)
        else:
            break

    return np.mean(angles)


def rotate_img(img: np.ndarray, angle: float) -> np.array:
    """
    Rotate image by angle
    :param img: image array
    :param angle: rotation angle
    :return: rotated image
    """
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def evaluate_angle(img: np.ndarray, angle: float) -> int:
    """
    Evaluate relevance of angle for image rotation
    :param img: image array
    :param angle: angle
    :return: metric for angle quality
    """
    # Rotate image
    rotated_img = rotate_img(img=img, angle=angle)
    # Apply horizontal projection
    proj = np.sum(rotated_img, 1)
    # Count number of empty rows
    return np.sum((proj[1:] - proj[:-1]) ** 2)


def estimate_skew(angles: List[float], thresh: np.ndarray) -> float:
    """
    Estimate skew from angles
    :param angles: list of angles
    :param thresh: thresholded image
    :return: best angle
    """
    # If there is only one angle, return it
    if len(angles) == 1:
        return angles.pop()

    if angles[-1] - angles[0] <= 0.015:
        # Get angle by applying Dixon Q test
        best_angle = angle_dixon_q_test(angles=angles)
    else:
        # Evaluate angles by rotation
        best_angle = None
        best_evaluation = 0
        for angle in sorted(angles, key=lambda a: abs(a)):
            # Get angle evaluation
            angle_evaluation = evaluate_angle(img=thresh, angle=angle)

            if angle_evaluation > best_evaluation:
                best_angle = angle
                best_evaluation = angle_evaluation

    return best_angle or 0


def rotate_img_with_border(img: np.ndarray, angle: float,
                           background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Rotate an image of the defined angle and add background on border
    :param img: image array
    :param angle: rotation angle
    :param background_color: background color for borders after rotation
    :return: rotated image array
    """
    # Compute image center
    height, width = img.shape
    image_center = (width // 2, height // 2)

    # Compute rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # Get rotated image dimension
    bound_w = int(height * abs(rotation_mat[0, 1]) + width * abs(rotation_mat[0, 0]))
    bound_h = int(height * abs(rotation_mat[0, 0]) + width * abs(rotation_mat[0, 1]))

    # Update rotation matrix
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # Create rotated image with white background
    rotated_img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=background_color)
    return rotated_img


def fix_rotation_image(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Fix rotation of input image (based on https://www.mdpi.com/2079-9292/9/1/55) by at most 45 degrees
    :param img: image array
    :return: rotated image array and boolean indicating if the image has been rotated
    """
    # Get connected components of the images
    cc_centroids, ref_height, thresh = get_connected_components(img=img)

    # Check number of centroids
    if len(cc_centroids) < 2:
        return img, False

    # Compute most likely angles from connected components
    angles = get_relevant_angles(centroids=cc_centroids, ref_height=ref_height)
    # Estimate skew
    skew_angle = estimate_skew(angles=angles, thresh=thresh)

    if abs(skew_angle) >= 0.25:
        # Rotate image with borders
        return rotate_img_with_border(img=img, angle=skew_angle), True

    return img, False



def cluster_items(items: List[Any], clustering_func: Callable) -> List[List[Any]]:
    """
    Cluster items based on a function
    :param items: list of items
    :param clustering_func: clustering function
    :return: list of list of items based on clustering function
    """
    # Create clusters based on clustering function between items
    clusters = list()
    for i in range(len(items)):
        for j in range(i, len(items)):
            # Check if both items corresponds according to the clustering function
            corresponds = clustering_func(items[i], items[j]) or (items[i] == items[j])

            # If both items correspond, find matching clusters or create a new one
            if corresponds:
                matching_clusters = [idx for idx, cl in enumerate(clusters) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(clusters) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(clusters) if idx in matching_clusters])
                    clusters = remaining_clusters + [new_cluster]
                else:
                    clusters.append({i, j})

    return [[items[idx] for idx in c] for c in clusters]



def is_contained_cell(inner_cell: Union[Cell, tuple], outer_cell: Union[Cell, tuple], percentage: float = 0.9) -> bool:
    """
    Assert if the inner cell is contained in outer cell
    :param inner_cell: inner cell
    :param outer_cell: Table object
    :param percentage: percentage of the inner cell that needs to be contained in the outer cell
    :return: boolean indicating if the inner cell is contained in the outer cell
    """
    # If needed, convert inner cell to Cell object
    if isinstance(inner_cell, tuple):
        inner_cell = Cell(*inner_cell)
    # If needed, convert outer cell to Cell object
    if isinstance(outer_cell, tuple):
        outer_cell = Cell(*outer_cell)

    # Compute common coordinates
    x_left = max(inner_cell.x1, outer_cell.x1)
    y_top = max(inner_cell.y1, outer_cell.y1)
    x_right = min(inner_cell.x2, outer_cell.x2)
    y_bottom = min(inner_cell.y2, outer_cell.y2)

    # Compute intersection area as well as inner cell area
    intersection_area = max(0, (x_right - x_left)) * max(0, (y_bottom - y_top))

    return intersection_area / inner_cell.area >= percentage


def merge_overlapping_contours(contours: List[Cell]) -> List[Cell]:
    """
    Merge overlapping contours
    :param contours: list of contours as Cell objects
    :return: list of merged contours
    """
    if len(contours) == 0:
        return []

    # Create dataframe with contours
    df_cnt = pl.LazyFrame(data=[{"id": idx, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2, "area": c.area}
                                for idx, c in enumerate(contours)])

    # Cross join
    df_cross = (df_cnt.join(df_cnt, how='cross')
                .filter(pl.col('id') != pl.col('id_right'))
                .filter(pl.col('area') <= pl.col('area_right'))
                )

    # Compute intersection area between contours and identify if the smallest contour overlaps the largest one
    x_left = pl.max_horizontal('x1', 'x1_right')
    x_right = pl.min_horizontal('x2', 'x2_right')
    y_top = pl.max_horizontal('y1', 'y1_right')
    y_bottom = pl.min_horizontal('y2', 'y2_right')
    intersection = pl.max_horizontal(x_right - x_left, 0) * pl.max_horizontal(y_bottom - y_top, 0)

    df_cross = (df_cross.with_columns(intersection.alias('intersection'))
                .with_columns((pl.col('intersection') / pl.col('area') >= 0.25).alias('overlaps'))
                )

    # Identify relevant contours: no contours is overlapping it
    deleted_contours = df_cross.filter(pl.col('overlaps')).select('id').unique()
    df_overlap = (df_cross.filter(pl.col('overlaps'))
                  .group_by(pl.col('id_right').alias('id'))
                  .agg(pl.min('x1').alias('x1_overlap'),
                       pl.max('x2').alias('x2_overlap'),
                       pl.min('y1').alias('y1_overlap'),
                       pl.max('y2').alias('y2_overlap'))
                  )

    df_final = (df_cnt.join(deleted_contours, on="id", how="anti")
                .join(df_overlap, on='id', how='left')
                .select([pl.min_horizontal('x1', 'x1_overlap').alias('x1'),
                         pl.max_horizontal('x2', 'x2_overlap').alias('x2'),
                         pl.min_horizontal('y1', 'y1_overlap').alias('y1'),
                         pl.max_horizontal('y2', 'y2_overlap').alias('y2'),
                         ])
                )

    # Map results to cells
    return [Cell(**d) for d in df_final.collect().to_dicts()]


def merge_contours(contours: List[Cell], vertically: Optional[bool] = True) -> List[Cell]:
    """
    Create merge contours by an axis
    :param contours: list of contours
    :param vertically: boolean indicating if contours are merged according to the vertical or horizontal axis
    :return: merged contours
    """
    # If contours is empty, return empty list
    if len(contours) == 0:
        return contours

    # If vertically is None, merge only contained contours
    if vertically is None:
        return merge_overlapping_contours(contours=contours)

    # Define dimensions used to merge contours
    idx_1 = "y1" if vertically else "x1"
    idx_2 = "y2" if vertically else "x2"
    sort_idx_1 = "x1" if vertically else "y1"
    sort_idx_2 = "x2" if vertically else "y2"

    # Sort contours
    sorted_cnts = sorted(contours,
                         key=lambda cnt: (getattr(cnt, idx_1), getattr(cnt, idx_2), getattr(cnt, sort_idx_1)))

    # Loop over contours and merge overlapping contours
    seq = iter(sorted_cnts)
    list_cnts = [copy.deepcopy(next(seq))]
    for cnt in seq:
        # If contours overlap, update current contour
        if getattr(cnt, idx_1) <= getattr(list_cnts[-1], idx_2):
            # Update current contour coordinates
            setattr(list_cnts[-1], idx_2, max(getattr(list_cnts[-1], idx_2), getattr(cnt, idx_2)))
            setattr(list_cnts[-1], sort_idx_1, min(getattr(list_cnts[-1], sort_idx_1), getattr(cnt, sort_idx_1)))
            setattr(list_cnts[-1], sort_idx_2, max(getattr(list_cnts[-1], sort_idx_2), getattr(cnt, sort_idx_2)))
        else:
            list_cnts.append(copy.deepcopy(cnt))

    return list_cnts


def get_contours_cell(img: np.ndarray, cell: Cell, margin: int = 5, blur_size: int = 9, kernel_size: int = 15,
                      merge_vertically: Optional[bool] = True) -> List[Cell]:
    """
    Get list of contours contained in cell
    :param img: image array
    :param cell: Cell object
    :param margin: margin in pixels used for cropped images
    :param blur_size: kernel size for blurring operation
    :param kernel_size: kernel size for dilate operation
    :param merge_vertically: boolean indicating if contours are merged according to the vertical or horizontal axis
    :return: list of contours contained in cell
    """
    height, width = img.shape[:2]
    # Get cropped image
    cropped_img = img[max(cell.y1 - margin, 0):min(cell.y2 + margin, height),
                      max(cell.x1 - margin, 0):min(cell.x2 + margin, width)]

    # If cropped image is empty, do not do anything
    height_cropped, width_cropped = cropped_img.shape[:2]
    if height_cropped <= 0 or width_cropped <= 0:
        return []

    # Reprocess images
    blur = cv2.GaussianBlur(cropped_img, (blur_size, blur_size), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    list_cnts_cell = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x = x + cell.x1 - margin
        y = y + cell.y1 - margin
        contour_cell = Cell(x, y, x + w, y + h)
        list_cnts_cell.append(contour_cell)

    # Add contours to row
    contours = merge_contours(contours=list_cnts_cell,
                              vertically=merge_vertically)

    return contours




def prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Prepare image by removing background and keeping a white base
    :param img: original image array
    :return: processed image
    """
    # Preprocess image
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dilation = cv2.dilate(thresh, (10, 10), iterations=3)

    # Compute contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get contours cells
    contour_cells = list()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        contour_cells.append(Cell(x, y, x + w, y + h))
    contour_cells = sorted(contour_cells, key=lambda c: c.area, reverse=True)

    if contour_cells:
        largest_contour = None
        if len(contour_cells) == 1:
            # Set largest contour
            largest_contour = contour_cells.pop(0)
        elif contour_cells[0].area / contour_cells[1].area > 10:
            # Set largest contour
            largest_contour = contour_cells.pop(0)

        if largest_contour:
            # Recreate image from blank image by adding largest contour of the original image
            processed_img = np.zeros(img.shape, dtype=np.uint8)
            processed_img.fill(255)

            # Add contour from original image
            cropped_img = img[largest_contour.y1:largest_contour.y2, largest_contour.x1:largest_contour.x2]
            processed_img[largest_contour.y1:largest_contour.y2, largest_contour.x1:largest_contour.x2] = cropped_img

            return processed_img

    return img




def threshold_dark_areas(img: np.ndarray, char_length: Optional[float]) -> np.ndarray:
    """
    Threshold image by differentiating areas with light and dark backgrounds
    :param img: image array
    :param char_length: average character length
    :return: threshold image
    """
    # Get threshold on image and binary image
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    thresh_kernel = max(int(round(char_length)), 1) if char_length else 21
    thresh_kernel = thresh_kernel + 1 if thresh_kernel % 2 == 0 else thresh_kernel

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_kernel, 5)
    binary_thresh = cv2.adaptiveThreshold(255 - blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_kernel, 5)

    # Mask on areas with dark background
    blur_size = min(255, max(int(2 * char_length) + 1 - int(2 * char_length) % 2, 1) if char_length else 11)
    blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    mask = cv2.inRange(blur, 0, 100)

    # Get contours of dark areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each dark area, use binary threshold instead of regular threshold
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        margin = int(char_length) if char_length else 21
        if min(w, h) > 2 * margin and w * h / np.prod(img.shape[:2]) < 0.9:
            thresh[y+margin:y+h-margin, x+margin:x+w-margin] = binary_thresh[y+margin:y+h-margin, x+margin:x+w-margin]

    return thresh


def dilate_dotted_lines(thresh: np.ndarray, char_length: float, contours: List[Cell]) -> np.ndarray:
    """
    Dilate specific rows/columns of the threshold image in order to detect dotted rows
    :param thresh: threshold image array
    :param char_length: average character length in image
    :param contours: list of image contours as cell objects
    :return: threshold image with dilated dotted rows
    """
    # Compute non-null thresh and its average value
    non_null_thresh = thresh[:, np.max(thresh, axis=0) > 0]
    non_null_thresh = non_null_thresh[np.max(non_null_thresh, axis=1) > 0, :]
    w_mean = np.mean(non_null_thresh)

    ### Horizontal case
    # Create dilated image
    h_dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(char_length), 1), 1)))

    # Get rows with at least 2 times the average number of white pixels
    h_non_null = np.where(np.max(thresh, axis=1) > 0)[0]
    white_rows = np.where(np.mean(thresh[:, min(h_non_null):max(h_non_null)], axis=1) > 4 * w_mean)[0].tolist()

    # Split into consecutive groups of rows and keep only small ones to avoid targeting text rows
    white_rows_cl = [list(map(itemgetter(1), g))
                     for k, g in groupby(enumerate(white_rows), lambda i_x: i_x[0] - i_x[1])]

    # Filter clusters with contours
    filtered_rows_cl = list()
    for row_cl in white_rows_cl:
        # Compute percentage of white pixels in rows
        pct_w_pixels = np.mean(thresh[row_cl, :]) / 255
        # Compute percentage of rows covered by contours
        covered_contours = [cnt for cnt in contours if min(cnt.y2, max(row_cl)) - max(cnt.y1, min(row_cl)) > 0]
        pct_contours = sum(map(lambda cnt: cnt.width, covered_contours)) / thresh.shape[1]

        if 0.66 * pct_w_pixels >= pct_contours:
            filtered_rows_cl.append(row_cl)

    white_rows_final = [idx for cl in filtered_rows_cl for idx in cl]

    # Keep only dilated image on specific rows
    mask = np.ones(thresh.shape[0], dtype=bool)
    mask[white_rows_final] = False
    h_dilated[mask, :] = 0

    ### Vertical case
    # Create dilated image
    v_dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(char_length), 1))))

    # Get columns with at least 2 times the average number of white pixels
    v_non_null = np.where(np.max(thresh, axis=0) > 0)[0]
    white_cols = np.where(np.mean(thresh[min(v_non_null):max(v_non_null), :], axis=0) > 4 * w_mean)[0].tolist()

    # Split into consecutive groups of columns and keep only small ones to avoid targeting text columns
    white_cols_cl = [list(map(itemgetter(1), g))
                     for k, g in groupby(enumerate(white_cols), lambda i_x: i_x[0] - i_x[1])]

    # Filter clusters with contours
    filtered_cols_cl = list()
    for col_cl in white_cols_cl:
        # Compute percentage of white pixels in columns
        pct_w_pixels = np.mean(thresh[:, col_cl]) / 255
        # Compute percentage of columns covered by contours
        covered_contours = [cnt for cnt in contours if min(cnt.x2, max(col_cl)) - max(cnt.x1, min(col_cl)) > 0]
        pct_contours = sum(map(lambda cnt: cnt.height, covered_contours)) / thresh.shape[0]

        if 0.66 * pct_w_pixels >= pct_contours:
            filtered_cols_cl.append(col_cl)

    white_cols_final = [idx for cl in filtered_cols_cl for idx in cl]

    # Keep only dilated image on specific columns
    mask = np.ones(thresh.shape[1], dtype=bool)
    mask[white_cols_final] = False
    v_dilated[:, mask] = 0

    # Update thresh
    new_thresh = np.maximum(thresh, h_dilated)
    new_thresh = np.maximum(new_thresh, v_dilated)

    return new_thresh


def overlapping_filter(lines: List[Line], max_gap: int = 5) -> List[Line]:
    """
    Process rows to merge close rows
    :param lines: rows
    :param max_gap: maximum gap used to merge rows
    :return: list of filtered rows
    """
    if len(lines) == 0:
        return []

    # Identify if rows are horizontal
    horizontal = np.average([l.horizontal for l in lines], weights=[l.length for l in lines]) > 0.5

    # If not horizontal, transpose all rows
    if not horizontal:
        lines = [line.transpose for line in lines]

    # Sort rows by secondary dimension
    lines = sorted(lines, key=lambda l: (l.y1, l.x1))

    # Create clusters of rows based on "similar" secondary dimension
    previous_sequence, current_sequence = iter(lines), iter(lines)
    line_clusters = [[next(current_sequence)]]
    for previous, line in zip(previous_sequence, current_sequence):
        # If the vertical difference between consecutive rows is too large, create a new cluster
        if line.y1 - previous.y1 > 2:
            # Large gap, we create a new empty sublist
            line_clusters.append([])

        # Append to last cluster
        line_clusters[-1].append(line)

    # Create final rows by "merging" rows within a cluster
    final_lines = list()
    for cluster in line_clusters:
        # Sort the cluster
        cluster = sorted(cluster, key=lambda l: min(l.x1, l.x2))

        # Loop over rows in the cluster to merge relevant rows together
        seq = iter(cluster)
        sub_clusters = [[next(seq)]]
        for line in seq:
            # If rows are vertically close, merge line with curr_line
            dim_2_sub_clust = max(map(lambda l: l.x2, sub_clusters[-1]))
            if line.x1 - dim_2_sub_clust <= max_gap:
                sub_clusters[-1].append(line)
            # If the difference in vertical coordinates is too large, create a new sub cluster
            else:
                sub_clusters.append([line])

        # Create rows from sub clusters
        for sub_cl in sub_clusters:
            y_value = int(round(np.average([l.y1 for l in sub_cl],
                                           weights=list(map(lambda l: l.length, sub_cl)))))
            thickness = min(max(1, max(map(lambda l: l.y2, sub_cl)) - min(map(lambda l: l.y1, sub_cl))), 5)
            line = Line(x1=min(map(lambda l: l.x1, sub_cl)),
                        x2=max(map(lambda l: l.x2, sub_cl)),
                        y1=int(y_value),
                        y2=int(y_value),
                        thickness=thickness)

            if line.length > 0:
                final_lines.append(line)

    # If not horizontal, transpose all rows
    if not horizontal:
        final_lines = [line.transpose for line in final_lines]

    return final_lines


def create_lines_from_intersection(line_dict: Dict) -> List[Line]:
    """
    Create list of lines from detected line and its intersecting elements
    :param line_dict: dictionary containing line and its intersecting elements
    :return: list of relevant line objects
    """
    # Get intersection segments
    inter_segs = [(inter_cnt.get('y1'), inter_cnt.get('y2')) if line_dict.get('vertical')
                  else (inter_cnt.get('x1'), inter_cnt.get('x2'))
                  for inter_cnt in line_dict.get('intersecting') or []
                  ]

    if len(inter_segs) == 0:
        # If no elements intersect the line, return it
        return [Line(x1=line_dict.get('x1_line'),
                     x2=line_dict.get('x2_line'),
                     y1=line_dict.get('y1_line'),
                     y2=line_dict.get('y2_line'),
                     thickness=line_dict.get('thickness'))
                ]
    
    # Vertical case
    if line_dict.get('vertical'):
        # Get x and y values of the line
        x, y_min, y_max = line_dict.get('x1_line'), line_dict.get('y1_line'), line_dict.get('y2_line')
        # Create y range of the line
        y_range = list(range(y_min, y_max + 1))

        # For each intersecting elements, remove common coordinates with the line
        for inter_seg in inter_segs:
            y_range = [y for y in y_range if not inter_seg[0] <= y <= inter_seg[1]]

        if y_range:
            # Create list of lists of consecutive y values from the range
            seq = iter(y_range)
            line_y_gps = [[next(seq)]]
            for y in seq:
                if y > line_y_gps[-1][-1] + 1:
                    line_y_gps.append([])
                line_y_gps[-1].append(y)

            return [Line(x1=x, x2=x, y1=min(y_gp), y2=max(y_gp), thickness=line_dict.get('thickness'))
                    for y_gp in line_y_gps]
        return []
    # Horizontal case
    else:
        # Get x and y values of the line
        y, x_min, x_max = line_dict.get('y1_line'), line_dict.get('x1_line'), line_dict.get('x2_line')
        # Create x range of the line
        x_range = list(range(x_min, x_max + 1))

        # For each intersecting elements, remove common coordinates with the line
        for inter_seg in inter_segs:
            x_range = [x for x in x_range if not inter_seg[0] <= x <= inter_seg[1]]

        if x_range:
            # Create list of lists of consecutive x values from the range
            seq = iter(x_range)
            line_x_gps = [[next(seq)]]
            for x in seq:
                if x > line_x_gps[-1][-1] + 1:
                    line_x_gps.append([])
                line_x_gps[-1].append(x)

            return [Line(y1=y, y2=y, x1=min(x_gp), x2=max(x_gp), thickness=line_dict.get('thickness'))
                    for x_gp in line_x_gps]
        return []


def remove_word_lines(lines: List[Line], contours: List[Cell]) -> List[Line]:
    """
    Remove rows that corresponds to contours in image
    :param lines: list of rows
    :param contours: list of image contours as cell objects
    :return: list of rows not intersecting with words
    """
    # If there are no rows or no contours, do nothing
    if len(lines) == 0 or len(contours) == 0:
        return lines

    # Get contours dataframe
    df_cnts = pl.LazyFrame(data=[{"x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2} for c in contours])

    # Create dataframe containing rows
    df_lines = (pl.LazyFrame(data=[{**line.dict, **{"id_line": idx}} for idx, line in enumerate(lines)])
                .with_columns([pl.max_horizontal([pl.col('width'), pl.col('height')]).alias('length'),
                               (pl.col('x1') == pl.col('x2')).alias('vertical')]
                              )
                .rename({"x1": "x1_line", "x2": "x2_line", "y1": "y1_line", "y2": "y2_line"})
                )

    # Merge both dataframes
    df_words_lines = df_cnts.join(df_lines, how='cross')

    # Compute intersection between contours bbox and rows
    # - vertical case
    vert_int = (
            (((pl.col('x1') + pl.col('x2')) / 2 - pl.col('x1_line')).abs() / (pl.col('x2') - pl.col('x1')) < 0.5)
            & ((pl.min_horizontal(['y2', 'y2_line']) - pl.max_horizontal(['y1', 'y1_line'])) > 0)
    )
    # - horizontal case
    hor_int = (
            (((pl.col('y1') + pl.col('y2')) / 2 - pl.col('y1_line')).abs() / (pl.col('y2') - pl.col('y1')) <= 0.4)
            & ((pl.min_horizontal(['x2', 'x2_line']) - pl.max_horizontal(['x1', 'x1_line'])) > 0)
    )
    
    df_words_lines = df_words_lines.with_columns(
        ((pl.col('vertical') & vert_int) | ((~pl.col('vertical')) & hor_int)).alias('intersection')
        )
    
    # Get lines together with elements that intersect the line
    line_elements = (df_words_lines.filter(pl.col('intersection'))
                     .group_by(["id_line", "x1_line", "y1_line", "x2_line", "y2_line", "vertical", "thickness"])
                     .agg(pl.struct("x1", "y1", "x2", "y2").alias('intersecting'))
                     .unique(subset=["id_line"])
                     .collect()
                     .to_dicts()
                     )

    # Create lines from line elements
    modified_lines = {el.get('id_line') for el in line_elements}
    kept_lines = [line for id_line, line in enumerate(lines) if id_line not in modified_lines]
    reprocessed_lines = [line for line_dict in line_elements
                         for line in create_lines_from_intersection(line_dict=line_dict)]

    return kept_lines + reprocessed_lines


def detect_lines(thresh: np.ndarray, contours: Optional[List[Cell]], char_length: Optional[float], rho: float = 1,
                 theta: float = np.pi / 180, threshold: int = 50, minLinLength: int = 290, maxLineGap: int = 6,
                 kernel_size: int = 20) -> (List[Line], List[Line]):
    """
    Detect horizontal and vertical rows on image
    :param thresh: thresholded image array
    :param contours: list of image contours as cell objects
    :param char_length: average character length
    :param rho: rho parameter for Hough line transform
    :param theta: theta parameter for Hough line transform
    :param threshold: threshold parameter for Hough line transform
    :param minLinLength: minLinLength parameter for Hough line transform
    :param maxLineGap: maxLineGap parameter for Hough line transform
    :param kernel_size: kernel size to filter on horizontal / vertical rows
    :return: horizontal and vertical rows
    """
    if char_length is not None:
        # Process threshold image in order to detect dotted rows
        thresh = dilate_dotted_lines(thresh=thresh, char_length=char_length, contours=contours)

    # Identify both vertical and horizontal rows
    for kernel_tup, gap in [((kernel_size, 1), 2 * maxLineGap), ((1, kernel_size), maxLineGap)]:
        # Apply masking on image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_tup)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Compute Hough rows on image and get rows
        hough_lines = cv2.HoughLinesP(mask, rho, theta, threshold, None, minLinLength, maxLineGap)

        # Handle case with no rows
        if hough_lines is None:
            yield []
            continue

        lines = [Line(*line[0].tolist()).reprocess() for line in hough_lines]

        # Remove rows that are not horizontal or vertical
        lines = [line for line in lines if line.horizontal or line.vertical]

        # Merge rows
        merged_lines = overlapping_filter(lines=lines, max_gap=gap)

        # If possible, remove rows that correspond to words
        if contours is not None:
            merged_lines = remove_word_lines(lines=merged_lines, contours=contours)
            merged_lines = [l for l in merged_lines if max(l.length, l.width) >= minLinLength]

        yield merged_lines


def get_cells(horizontal_lines: List[Line], vertical_lines: List[Line]) -> List[Cell]:
    """
    Identify cells from horizontal and vertical rows
    :param horizontal_lines: list of horizontal rows
    :param vertical_lines: list of vertical rows
    :return: list of all cells in image
    """
    # Create dataframe with cells from horizontal and vertical rows
    df_cells = get_cells_dataframe(horizontal_lines=horizontal_lines,
                                   vertical_lines=vertical_lines)

    # Handle case of empty cells
    if df_cells.collect().height == 0:
        return []

    # Deduplicate cells
    df_cells_dedup = deduplicate_cells(df_cells=df_cells)

    # Convert to Cell objects
    cells = [Cell(x1=row["x1"], x2=row["x2"], y1=row["y1"], y2=row["y2"])
             for row in df_cells_dedup.collect().to_dicts()]

    return cells




def deduplicate_cells(df_cells: pl.LazyFrame) -> pl.LazyFrame:
    """
    Deduplicate nested cells in order to keep the smallest ones
    :param df_cells: dataframe containing cells
    :return: dataframe containing cells after deduplication of the nested ones
    """
    # Create columns corresponding to cell characteristics
    df_cells = (df_cells.with_columns([(pl.col('x2') - pl.col('x1')).alias('width'),
                                       (pl.col('y2') - pl.col('y1')).alias('height')])
                .with_columns((pl.col('height') * pl.col('width')).alias('area'))
                )

    # Create copy of df_cells
    df_cells_cp = (df_cells.clone()
                   .rename({col: f"{col}_" for col in df_cells.columns})
                   )

    if df_cells.collect().height == 0:
        return df_cells

    # Cross join to get cells pairs and filter on right cells bigger than right cells
    df_cross_cells = (df_cells.clone()
                      .join(df_cells_cp, how='cross')
                      .filter(pl.col('index') != pl.col('index_'))
                      .filter(pl.col('area') <= pl.col('area_'))
                      )

    ### Compute indicator if the first cell is contained in second cell
    # Compute coordinates of intersection
    df_cross_cells = df_cross_cells.with_columns([pl.max_horizontal(['x1', 'x1_']).alias('x_left'),
                                                  pl.max_horizontal(['y1', 'y1_']).alias('y_top'),
                                                  pl.min_horizontal(['x2', 'x2_']).alias('x_right'),
                                                  pl.min_horizontal(['y2', 'y2_']).alias('y_bottom'),
                                                  ])

    # Compute area of intersection
    df_cross_cells = df_cross_cells.with_columns((pl.max_horizontal([pl.col('x_right') - pl.col('x_left'), pl.lit(0)])
                                                  * pl.max_horizontal([pl.col('y_bottom') - pl.col('y_top'), pl.lit(0)])
                                                  ).alias('int_area')
                                                 )

    # Create column indicating if left cell is contained in right cell
    df_cross_cells = df_cross_cells.with_columns(((pl.col('x_right') >= pl.col('x_left'))
                                                  & (pl.col('y_bottom') >= pl.col('y_top'))
                                                  & (pl.col('int_area') / pl.col('area') >= 0.9)
                                                  ).alias('contained')
                                                 )

    ### Compute indicator if cells are adjacent
    # Compute intersections and horizontal / vertical differences
    df_cross_cells = (df_cross_cells
                      .with_columns([(pl.col('x_right') - pl.col('x_left')).alias('overlapping_x'),
                                     (pl.col('y_bottom') - pl.col('y_top')).alias('overlapping_y')])
                      .with_columns(pl.min_horizontal([(pl.col(_1) - pl.col(_2)).abs()
                                                       for _1, _2 in itertools.product(['x1', 'x2'], ['x1_', 'x2_'])]
                                                      ).alias('diff_x'))
                      .with_columns(pl.min_horizontal([(pl.col(_1) - pl.col(_2)).abs()
                                                       for _1, _2 in itertools.product(['y1', 'y2'], ['y1_', 'y2_'])]
                                                      ).alias('diff_y'))
                      )

    # Create column indicating if both cells are adjacent and  column indicating if the right cell is redundant with
    # the left cell
    condition_adjacent = (((pl.col("overlapping_y") > 5) & (pl.col("diff_x") == 0))
                          | ((pl.col("overlapping_x") > 5) & (pl.col("diff_y") == 0))
                          )
    df_cross_cells = (df_cross_cells.with_columns(condition_adjacent.alias('adjacent'))
                      .with_columns((pl.col('contained') & pl.col('adjacent')).alias('redundant'))
                      )

    # Get list of redundant cells and remove them from original cell dataframe
    redundant_cells = (df_cross_cells.filter(pl.col('redundant'))
                       .collect()
                       .get_column('index_')
                       .unique()
                       .to_list()
                       )
    df_final_cells = (df_cells.with_row_count(name="cnt")
                      .filter(~pl.col('cnt').is_in(redundant_cells))
                      .drop('cnt')
                      )

    return df_final_cells



def get_potential_cells_from_h_lines(df_h_lines: pl.LazyFrame) -> pl.LazyFrame:
    """
    Identify potential cells by matching corresponding horizontal rows
    :param df_h_lines: dataframe containing horizontal rows
    :return: dataframe containing potential cells
    """
    # Create copy of df_h_lines
    df_h_lines_cp = (df_h_lines.clone()
                     .rename({col: f"{col}_" for col in df_h_lines.columns})
                     )

    # Cross join with itself to get pairs of horizontal rows
    cross_h_lines = (df_h_lines.join(df_h_lines_cp, how='cross')
                     .filter(pl.col('y1') < pl.col('y1_'))
                     )

    # Compute horizontal correspondences between rows
    cross_h_lines = cross_h_lines.with_columns([
        (((pl.col('x1') - pl.col('x1_')) / pl.col('width')).abs() <= 0.02).alias("l_corresponds"),
        (((pl.col('x2') - pl.col('x2_')) / pl.col('width')).abs() <= 0.02).alias("r_corresponds"),
        (((pl.col('x1') <= pl.col('x1_')) & (pl.col('x1_') <= pl.col('x2')))
         | ((pl.col('x1_') <= pl.col('x1')) & (pl.col('x1') <= pl.col('x2_')))).alias('l_contained'),
        (((pl.col('x1') <= pl.col('x2_')) & (pl.col('x2_') <= pl.col('x2')))
         | ((pl.col('x1_') <= pl.col('x2')) & (pl.col('x2') <= pl.col('x2_')))).alias('r_contained')
    ])

    # Create condition on horizontal correspondence in order to use both rows and filter on relevant combinations
    matching_condition = ((pl.col('l_corresponds') | pl.col('l_contained'))
                          & (pl.col('r_corresponds') | pl.col('r_contained')))
    cross_h_lines = cross_h_lines.filter(matching_condition)

    # Create cell bbox from horizontal rows
    df_bbox = (cross_h_lines.select([pl.max_horizontal(['x1', 'x1_']).alias('x1_bbox'),
                                     pl.min_horizontal(['x2', 'x2_']).alias('x2_bbox'),
                                     pl.col('y1').alias("y1_bbox"),
                                     pl.col('y1_').alias('y2_bbox')]
                                    )
               .with_row_count(name="idx")
               )

    # Deduplicate on upper bound
    df_bbox = (df_bbox.sort(by=["x1_bbox", "x2_bbox", "y1_bbox", "y2_bbox"])
               .with_columns(pl.lit(1).alias('ones'))
               .with_columns(pl.col('ones').cum_sum().over(["x1_bbox", "x2_bbox", "y1_bbox"]).alias('cell_rk'))
               .filter(pl.col('cell_rk') == 1)
               )

    # Deduplicate on lower bound
    df_bbox = (df_bbox.sort(by=["x1_bbox", "x2_bbox", "y2_bbox", "y1_bbox"], descending=[False, False, False, True])
               .with_columns(pl.lit(1).alias('ones'))
               .with_columns(pl.col('ones').cum_sum().over(["x1_bbox", "x2_bbox", "y2_bbox"]).alias('cell_rk'))
               .filter(pl.col('cell_rk') == 1)
               .drop(['ones', 'cell_rk'])
               )

    return df_bbox


def get_cells_dataframe(horizontal_lines: List[Line], vertical_lines: List[Line]) -> pl.LazyFrame:
    """
    Create dataframe of all possible cells from horizontal and vertical rows
    :param horizontal_lines: list of horizontal rows
    :param vertical_lines: list of vertical rows
    :return: dataframe containing all cells
    """
    # Check for empty rows
    if len(horizontal_lines) * len(vertical_lines) == 0:
        return pl.DataFrame().lazy()

    # Create dataframe from horizontal and vertical rows
    df_h_lines = pl.LazyFrame(data=[l.dict for l in horizontal_lines])
    df_v_lines = pl.LazyFrame(data=[l.dict for l in vertical_lines])

    # Identify potential cells bboxes from horizontal rows
    df_bbox = get_potential_cells_from_h_lines(df_h_lines=df_h_lines)

    # Cross join with vertical rows
    df_bbox = df_bbox.with_columns(pl.max_horizontal([(pl.col('x2_bbox') - pl.col('x1_bbox')) * 0.025,
                                                      pl.lit(5.0)]).round(0).alias('h_margin')
                                   )
    df_bbox_v = df_bbox.join(df_v_lines, how='cross')

    # Check horizontal correspondence between cell and vertical rows
    horizontal_cond = ((pl.col("x1_bbox") - pl.col("h_margin") <= pl.col("x1"))
                       & (pl.col("x2_bbox") + pl.col("h_margin") >= pl.col("x1")))
    df_bbox_v = df_bbox_v.filter(horizontal_cond)

    # Check vertical overlapping
    df_bbox_v = (df_bbox_v.with_columns((pl.min_horizontal(['y2', 'y2_bbox'])
                                         - pl.max_horizontal(['y1', 'y1_bbox'])).alias('overlapping')
                                        )
                 .filter(pl.col('overlapping') / (pl.col('y2_bbox') - pl.col('y1_bbox')) >= 0.8)
                 )

    # Get all vertical delimiters by bbox
    df_bbox_delimiters = (df_bbox_v.sort(['idx', "x1_bbox", "x2_bbox", "y1_bbox", "y2_bbox", "x1"])
                          .group_by(['idx', "x1_bbox", "x2_bbox", "y1_bbox", "y2_bbox"])
                          .agg(pl.col('x1').alias('dels'))
                          .filter(pl.col("dels").list.len() >= 2)
                          )

    # Create new cells based on vertical delimiters
    df_cells = (df_bbox_delimiters.explode("dels")
                .with_columns([pl.col('dels').shift(1).over(pl.col('idx')).alias("x1_bbox"),
                               pl.col('dels').alias("x2_bbox")])
                .filter(pl.col('x1_bbox').is_not_null())
                .select([pl.col("x1_bbox").alias("x1"),
                         pl.col("y1_bbox").alias("y1"),
                         pl.col("x2_bbox").alias("x2"),
                         pl.col("y2_bbox").alias("y2")
                         ])
                .sort(['x1', 'y1', 'x2', 'y2'])
                .with_row_count(name="index")
                )

    return df_cells



def get_tables(cells: List[Cell], elements: List[Cell], lines: List[Line], char_length: float) -> List[Table]:
    """
    Identify and create Table object from list of image cells
    :param cells: list of cells found in image
    :param elements: list of image elements
    :param lines: list of image lines
    :param elements: average character length
    :return: list of Table objects inferred from cells
    """
    # Cluster cells into tables
    list_cluster_cells = cluster_cells_in_tables(cells=cells)

    # Normalize cells in clusters
    clusters_normalized = [normalize_table_cells(cluster_cells=cluster_cells)
                           for cluster_cells in list_cluster_cells]

    # Add semi-bordered cells to clusters
    complete_clusters = [add_semi_bordered_cells(cluster=cluster, lines=lines, char_length=char_length)
                         for cluster in clusters_normalized]

    # Create tables from cells clusters
    tables = [cluster_to_table(cluster_cells=cluster, elements=elements)
              for cluster in complete_clusters]

    return [tb for tb in tables if tb.nb_rows * tb.nb_columns >= 2]



def get_adjacent_cells(cells: List[Cell]) -> List[Set[int]]:
    """
    Identify adjacent cells
    :param cells: list of cells
    :return: list of sets of adjacent cells indexes
    """
    if len(cells) == 0:
        return []

    df_cells = pl.LazyFrame([{"idx": idx, "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2, "height": c.height,
                              "width": c.width}
                             for idx, c in enumerate(cells)])

    # Crossjoin and identify adjacent cells
    df_adjacent_cells = (
        df_cells.join(df_cells, how='cross')
        # Compute horizontal and vertical overlap
        .with_columns((pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right'])).alias("x_overlap"),
                      (pl.min_horizontal(['y2', 'y2_right']) - pl.max_horizontal(['y1', 'y1_right'])).alias("y_overlap")
                      )
        # Compute horizontal and vertical differences
        .with_columns(
            pl.min_horizontal((pl.col('x1') - pl.col('x1_right')).abs(),
                              (pl.col('x1') - pl.col('x2_right')).abs(),
                              (pl.col('x2') - pl.col('x1_right')).abs(),
                              (pl.col('x2') - pl.col('x2_right')).abs()
                              ).alias('diff_x'),
            pl.min_horizontal((pl.col('y1') - pl.col('y1_right')).abs(),
                              (pl.col('y1') - pl.col('y2_right')).abs(),
                              (pl.col('y2') - pl.col('y1_right')).abs(),
                              (pl.col('y2') - pl.col('y2_right')).abs()
                              ).alias('diff_y')
        )
        # Compute thresholds for horizontal and vertical differences
        .with_columns(
            pl.min_horizontal(pl.lit(5), 0.05 * pl.min_horizontal(pl.col('width'), pl.col('width_right'))).alias('thresh_x'),
            pl.min_horizontal(pl.lit(5), 0.05 * pl.min_horizontal(pl.col('height'), pl.col('height_right'))).alias('thresh_y')
        )
        # Filter adjacent cells
        .filter(
           ((pl.col('y_overlap') > 5) & (pl.col('diff_x') <= pl.col('thresh_x')))
            | ((pl.col('x_overlap') > 5) & (pl.col('diff_y') <= pl.col('thresh_y')))
        )
        .select("idx", "idx_right")
        .unique()
        .sort(by=['idx', 'idx_right'])
        .collect()
    )

    # Get sets of adjacent cells indexes
    adjacent_cells = [{row.get('idx'), row.get('idx_right')} for row in df_adjacent_cells.to_dicts()]

    return adjacent_cells


def cluster_cells_in_tables(cells: List[Cell]) -> List[List[Cell]]:
    """
    Based on adjacent cells, create clusters of cells that corresponds to tables
    :param cells: list cells in image
    :return: list of list of cells, representing several clusters of cells that form a table
    """
    # Get couples of adjacent cells
    adjacent_cells = get_adjacent_cells(cells=cells)

    # Loop over couples to create clusters
    clusters = find_components(edges=adjacent_cells)

    # Return list of cell objects
    list_table_cells = [[cells[idx] for idx in cl] for cl in clusters]

    return list_table_cells


def compute_table_median_row_sep(table: Table, contours: List[Cell]) -> Optional[float]:
    """
    Compute median row separation in table
    :param table: Table object
    :param contours: list of image contours as cell objects
    :return: median row separation
    """
    # Create dataframe with contours
    list_elements = [{"id": idx, "x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2}
                     for idx, el in enumerate(contours)]
    df_elements = pl.LazyFrame(data=list_elements)

    # Filter on elements that are within the table
    df_elements_table = df_elements.filter((pl.col('x1') >= table.x1) & (pl.col('x2') <= table.x2)
                                           & (pl.col('y1') >= table.y1) & (pl.col('y2') <= table.y2))

    # Cross join to get corresponding elements and filter on elements that corresponds horizontally
    df_h_elms = (df_elements_table.join(df_elements_table, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right']) > 0)
                 )

    # Get element which is directly below
    df_elms_below = (df_h_elms.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cum_sum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_elms_below.collect().height == 0:
        return None

    # Compute median vertical distance between elements
    median_v_dist = (df_elms_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                  - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect()
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    return median_v_dist


def handle_implicit_rows_table(img: np.ndarray, table: Table, contours: List[Cell], margin: int = 5) -> Table:
    """
    Find implicit rows and update tables based on those
    :param img: image array
    :param table: Table object
    :param contours: list of image contours as cell objects
    :param lines: list of lines in image
    :param margin: margin in pixels used for cropped images
    :return: reprocessed table with implicit rows
    """
    height, width = img.shape[:2]

    # If table is a single cell, do not search for implicit rows
    if table.nb_columns * table.nb_rows <= 1:
        return table

    # Get median row separation
    median_row_sep = compute_table_median_row_sep(table=table, contours=contours)

    if median_row_sep is None:
        return table

    list_splitted_rows = list()
    # Check if each row can be splitted
    for row in table.items:
        # If row is not vertically consistent, it is not relevant to split it
        if not row.v_consistent:
            list_splitted_rows.append(row)
            continue

        # Get cropped image
        cropped_img = img[max(row.y1 - margin, 0):min(row.y2 + margin, height),
                          max(row.x1 - margin, 0):min(row.x2 + margin, width)]

        # If cropped image is empty, do not do anything
        height_cropped, width_cropped = cropped_img.shape[:2]
        if height_cropped <= 0 or width_cropped <= 0:
            list_splitted_rows.append(row)
            continue

        # Reprocess images
        blur = cv2.GaussianBlur(cropped_img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

        # Dilate to combine adjacent text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(median_row_sep // 3))))
        dilate = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Get list of contours
        list_cnts_cell = list()
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            x = x + row.x1 - margin
            y = y + row.y1 - margin
            contour_cell = Cell(x, y, x + w, y + h)
            list_cnts_cell.append(contour_cell)

        # Add contours to row
        row_cnts = merge_contours(contours=list_cnts_cell,
                                  vertically=True)

        # Delete contours that do not contains any elements
        filtered_contours = list()
        for row_cnt in row_cnts:
            # Get matching lines
            matching_els = [cnt for cnt in contours
                            if is_contained_cell(inner_cell=cnt, outer_cell=row_cnt, percentage=0.8)]

            if len(matching_els) == 0:
                continue
            filtered_contours.append(row_cnt)

        # Compute vertical delimiters
        vertical_delimiters = sorted([int(round((cnt_1.y2 + cnt_2.y1) / 2))
                                      for cnt_1, cnt_2 in zip(filtered_contours, filtered_contours[1:])])

        # Split row into multiple rows from vertical delimiters
        list_splitted_rows += row.split_in_rows(vertical_delimiters=vertical_delimiters)

    return Table(rows=list_splitted_rows)


def handle_implicit_rows(img: np.ndarray, tables: List[Table], contours: List[Cell]) -> List[Table]:
    """
    Detect and handle implicit rows in image tables
    :param img: image array
    :param tables: list of Table objects
    :param contours: list of image contours as cell objects
    :param lines: list of lines in image
    :return: list of Table objects updated taking into account implicit rows
    """
    # Detect implicit rows
    tables_implicit_rows = [handle_implicit_rows_table(img=img,
                                                       table=table,
                                                       contours=contours)
                            for table in tables]

    return tables_implicit_rows



def add_semi_bordered_cells(cluster: List[Cell], lines: List[Line], char_length: float):
    """
    Identify and add semi-bordered cells to cluster
    :param cluster: cluster of cells
    :param lines: lines in image
    :param char_length: average character length
    :return: cluster with add semi-bordered cells
    """
    # Compute cluster coordinates
    x_min, x_max = min([c.x1 for c in cluster]), max([c.x2 for c in cluster])
    y_min, y_max = min([c.y1 for c in cluster]), max([c.y2 for c in cluster])

    # Initialize new coordinates
    new_x_min, new_x_max, new_y_min, new_y_max = x_min, x_max, y_min, y_max

    # Find horizontal lines of the cluster
    y_values_cl = {c.y1 for c in cluster}.union({c.y2 for c in cluster})
    h_lines = [line for line in lines if line.horizontal
               and min(line.x2, x_max) - max(line.x1, x_min) >= 0.8 * (x_max - x_min)
               and min([abs(line.y1 - y) for y in y_values_cl]) <= 0.05 * (y_max - y_min)]

    # Check that all horizontal lines are coherent on the left end
    if all([abs(l1.x1 - l2.x1) <= 0.05 * np.mean([l1.length, l2.length])
            for l1 in h_lines for l2 in h_lines]) and len(h_lines) > 0:
        min_x_lines = max([l.x1 for l in h_lines])
        # Update table boundaries with lines
        new_x_min = min_x_lines if x_min - min_x_lines >= 2 * char_length else x_min

    # Check that all horizontal lines are coherent on the right end
    if all([abs(l1.x2 - l2.x2) <= 0.05 * np.mean([l1.length, l2.length])
            for l1 in h_lines for l2 in h_lines]) and len(h_lines) > 0:
        max_x_lines = min([l.x2 for l in h_lines])
        # Update table boundaries with lines
        new_x_max = max_x_lines if max_x_lines - x_max >= 2 * char_length else x_max

    # Find vertical lines of the cluster
    x_values_cl = {c.x1 for c in cluster}.union({c.x2 for c in cluster})
    v_lines = [line for line in lines if line.vertical
               and min(line.y2, y_max) - max(line.y1, y_min) >= 0.8 * (y_max - y_min)
               and min([abs(line.x1 - x) for x in x_values_cl]) <= 0.05 * (x_max - x_min)]

    # Check that all vertical lines are coherent on the top end
    if all([abs(l1.y1 - l2.y1) <= 0.05 * np.mean([l1.length, l2.length])
            for l1 in v_lines for l2 in v_lines]) and len(v_lines) > 0:
        min_y_lines = max([l.y1 for l in v_lines])
        # Update table boundaries with lines
        new_y_min = min_y_lines if y_min - min_y_lines >= 2 * char_length else y_min

    # Check that all vertical lines are coherent on the bottom end
    if all([abs(l1.y2 - l2.y2) <= 0.05 * np.mean([l1.length, l2.length])
            for l1 in v_lines for l2 in v_lines]) and len(v_lines) > 0:
        max_y_lines = min([l.y2 for l in v_lines])
        # Update table boundaries with lines
        new_y_max = max_y_lines if max_y_lines - y_max >= 2 * char_length else y_max

    if (x_min, x_max, y_min, y_max) == (new_x_min, new_x_max, new_y_min, new_y_max):
        return cluster

    # Create new cells
    new_y_values = sorted(list(y_values_cl.union({new_y_min, new_y_max})))
    new_x_values = sorted(list(x_values_cl.union({new_x_min, new_x_max})))

    left_cells = [Cell(x1=new_x_min, x2=x_min, y1=y_top, y2=y_bottom)
                  for y_top, y_bottom in zip(new_y_values, new_y_values[1:])]
    right_cells = [Cell(x1=x_max, x2=new_x_max, y1=y_top, y2=y_bottom)
                   for y_top, y_bottom in zip(new_y_values, new_y_values[1:])]
    top_cells = [Cell(x1=x_left, x2=x_right, y1=new_y_min, y2=y_min)
                 for x_left, x_right in zip(new_x_values, new_x_values[1:])]
    bottom_cells = [Cell(x1=x_left, x2=x_right, y1=y_max, y2=new_y_max)
                    for x_left, x_right in zip(new_x_values, new_x_values[1:])]

    # Update cluster cells
    cluster_cells = {c for c in cluster + left_cells + right_cells + top_cells + bottom_cells if c.area > 0}

    return list(cluster_cells)



def normalize_table_cells(cluster_cells: List[Cell]) -> List[Cell]:
    """
    Normalize cells from table cells
    :param cluster_cells: list of cells that form a table
    :return: list of normalized cells
    """
    # Compute table shape
    width = max(map(lambda c: c.x2, cluster_cells)) - min(map(lambda c: c.x1, cluster_cells))
    height = max(map(lambda c: c.y2, cluster_cells)) - min(map(lambda c: c.y1, cluster_cells))

    # Get list of existing horizontal values
    h_values = sorted(list(set([x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]])))
    # Compute delimiters by grouping close values together
    h_delims = [int(round(np.mean(h_group))) for h_group in
                np.split(h_values, np.where(np.diff(h_values) >= min(width * 0.02, 10))[0] + 1)]

    # Get list of existing vertical values
    v_values = sorted(list(set([y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]])))
    # Compute delimiters by grouping close values together
    v_delims = [int(round(np.mean(v_group))) for v_group in
                np.split(v_values, np.where(np.diff(v_values) >= min(height * 0.02, 10))[0] + 1)]

    # Normalize all cells
    normalized_cells = list()
    for cell in cluster_cells:
        normalized_cell = Cell(x1=sorted(h_delims, key=lambda d: abs(d - cell.x1)).pop(0),
                               x2=sorted(h_delims, key=lambda d: abs(d - cell.x2)).pop(0),
                               y1=sorted(v_delims, key=lambda d: abs(d - cell.y1)).pop(0),
                               y2=sorted(v_delims, key=lambda d: abs(d - cell.y2)).pop(0))
        # Check if cell is not empty
        if cell.area > 0:
            normalized_cells.append(normalized_cell)

    return normalized_cells


def remove_unwanted_elements(table: Table, elements: List[Cell]) -> Table:
    """
    Remove empty/unnecessary rows and columns from the table, based on elements
    :param table: input Table object
    :param elements: list of image elements
    :return: processed table
    """
    # Identify elements corresponding to each cell
    df_elements = pl.LazyFrame([{"x1_el": el.x1, "y1_el": el.y1, "x2_el": el.x2, "y2_el": el.y2, "area_el": el.area}
                                for el in elements])
    df_cells = pl.LazyFrame([{"id_row": id_row, "id_col": id_col,  "x1": c.x1, "y1": c.y1, "x2": c.x2, "y2": c.y2}
                             for id_row, row in enumerate(table.items)
                             for id_col, c in enumerate(row.items)])
    df_cells_elements = (
        df_cells.join(df_elements, how="cross")
        .with_columns((pl.min_horizontal(['x2', 'x2_el']) - pl.max_horizontal(['x1', 'x1_el'])).alias("x_overlap"),
                      (pl.min_horizontal(['y2', 'y2_el']) - pl.max_horizontal(['y1', 'y1_el'])).alias("y_overlap"))
        .filter(pl.col('x_overlap') > 0,
                pl.col('y_overlap') > 0)
        .with_columns((pl.col('x_overlap') * pl.col('y_overlap')).alias('area_intersection'))
        .filter(pl.col('area_intersection') / pl.col('area_el') >= 0.6)
        .select("id_row", "id_col")
        .unique()
        .collect()
    )

    # Identify empty rows and empty columns
    empty_rows = [id_row for id_row in range(table.nb_rows)
                  if id_row not in [rec.get('id_row') for rec in df_cells_elements.to_dicts()]]
    empty_cols = [id_col for id_col in range(table.nb_columns)
                  if id_col not in [rec.get('id_col') for rec in df_cells_elements.to_dicts()]]

    # Remove empty rows and empty columns
    table.remove_rows(row_ids=empty_rows)
    table.remove_columns(col_ids=empty_cols)

    return table


def cluster_to_table(cluster_cells: List[Cell], elements: List[Cell], borderless: bool = False) -> Table:
    """
    Convert a cell cluster to a Table object
    :param cluster_cells: list of cells that form a table
    :param elements: list of image elements
    :param borderless: boolean indicating if the created table is borderless
    :return: table with rows inferred from table cells
    """
    # Get list of vertical delimiters
    v_delims = sorted(list(set([y_val for cell in cluster_cells for y_val in [cell.y1, cell.y2]])))

    # Get list of horizontal delimiters
    h_delims = sorted(list(set([x_val for cell in cluster_cells for x_val in [cell.x1, cell.x2]])))

    # Create rows and cells
    list_rows = list()
    for y_top, y_bottom in zip(v_delims, v_delims[1:]):
        # Get matching cell
        matching_cells = [c for c in cluster_cells
                          if min(c.y2, y_bottom) - max(c.y1, y_top) >= 0.9 * (y_bottom - y_top)]
        list_cells = list()
        for x_left, x_right in zip(h_delims, h_delims[1:]):
            # Create default cell
            default_cell = Cell(x1=x_left, y1=y_top, x2=x_right, y2=y_bottom)

            # Check cells that contain the default cell
            containing_cells = sorted([c for c in matching_cells
                                       if is_contained_cell(inner_cell=default_cell, outer_cell=c, percentage=0.9)],
                                      key=lambda c: c.area)

            # Append either a cell that contain the default cell
            if containing_cells:
                list_cells.append(containing_cells.pop(0))
            else:
                # Get x value of closest matching cells
                x_value = sorted([x_val for cell in matching_cells for x_val in [cell.x1, cell.x2]],
                                 key=lambda x: min(abs(x - x_left), abs(x - x_right))).pop(0)
                list_cells.append(Cell(x1=x_value, y1=y_top, x2=x_value, y2=y_bottom))

        list_rows.append(Row(cells=list_cells))

    # Create table
    table = Table(rows=list_rows, borderless=borderless)

    # Remove empty/unnecessary rows and columns from the table, based on elements
    processed_table = remove_unwanted_elements(table=table, elements=elements)

    return processed_table

def deduplicate_tables(identified_tables: List[Table], existing_tables: List[Table]) -> List[Table]:
    """
    Deduplicate identified borderless tables with already identified tables in order to avoid duplicates and overlap
    :param identified_tables: list of borderless tables identified
    :param existing_tables: list of already identified tables
    :return: deduplicated list of identified borderless tables
    """
    # Sort tables by area
    identified_tables = sorted(identified_tables, key=lambda tb: tb.area, reverse=True)

    # For each table check if it does not overlap with an existing table
    final_tables = list()
    for table in identified_tables:
        if not any([max(is_contained_cell(inner_cell=table.cell, outer_cell=tb.cell, percentage=0.1),
                        is_contained_cell(inner_cell=tb.cell, outer_cell=table.cell, percentage=0.1))
                    for tb in existing_tables + final_tables]):
            final_tables.append(table)

    return final_tables


def identify_borderless_tables(thresh: np.ndarray, lines: List[Line], char_length: float, median_line_sep: float,
                               contours: List[Cell], existing_tables: List[Table]) -> List[Table]:
    """
    Identify borderless tables in image
    :param thresh: thresholded image array
    :param lines: list of rows detected in image
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param contours: list of image contours
    :param existing_tables: list of detected bordered tables
    :return: list of detected borderless tables
    """
    # Segment image and identify parts that can correspond to tables
    table_segments = segment_image(thresh=thresh,
                                   lines=lines,
                                   char_length=char_length,
                                   median_line_sep=median_line_sep)

    # In each segment, create groups of rows and identify tables
    tables = list()
    for table_segment in table_segments:
        # Identify column groups in segment
        column_group = identify_column_groups(table_segment=table_segment,
                                              char_length=char_length,
                                              median_line_sep=median_line_sep)

        if column_group:
            # Identify potential table rows
            table_rows = detect_delimiter_group_rows(delimiter_group=column_group)

            if table_rows:
                # Create table from column group and rows
                borderless_table = identify_table(columns=column_group,
                                                  table_rows=table_rows,
                                                  contours=contours,
                                                  median_line_sep=median_line_sep,
                                                  char_length=char_length)

                if borderless_table:
                    tables.append(borderless_table)

    return deduplicate_tables(identified_tables=tables,
                              existing_tables=existing_tables)






def get_whitespaces(segment: Union[ImageSegment, DelimiterGroup], vertical: bool = True, min_width: float = 0,
                    pct: float = 0.25) -> List[Cell]:
    """
    Identify whitespaces in segment
    :param segment: image segment
    :param vertical: boolean indicating if vertical or horizontal whitespaces are identified
    :param pct: minimum percentage of the segment height/width to account for a whitespace
    :param min_width: minimum width of the detected whitespaces
    :return: list of vertical or horizontal whitespaces as Cell objects
    """
    # Flip object coordinates in horizontal case
    if not vertical:
        flipped_elements = [Cell(x1=el.y1, y1=el.x1, x2=el.y2, y2=el.x2) for el in segment.elements]
        segment = ImageSegment(x1=segment.y1,
                               y1=segment.x1,
                               x2=segment.y2,
                               y2=segment.x2,
                               elements=flipped_elements)

    # Get min/max height of elements in segment
    y_min, y_max = min([el.y1 for el in segment.elements]), max([el.y2 for el in segment.elements])

    # Create dataframe containing elements
    df_elements = pl.concat(
        [pl.LazyFrame([{"x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2} for el in segment.elements]),
         pl.LazyFrame([{"x1": segment.x1, "y1": y, "x2": segment.x2, "y2": y} for y in [y_min, y_max]])]
    )

    # Get dataframe with relevant ranges
    df_x_ranges = (pl.concat([df_elements.select(pl.col('x1').alias('x')), df_elements.select(pl.col('x2').alias('x'))])
                   .unique()
                   .sort(by="x")
                   .select(pl.col('x').alias('x_min'), pl.col('x').shift(-1).alias('x_max'))
                   .filter(pl.col('x_max') - pl.col('x_min') >= min_width)
                   )

    # Get all elements within range and identify whitespaces
    df_elements_ranges = (
        df_x_ranges.join(df_elements, how='cross')
        .with_columns((pl.min_horizontal(pl.col('x_max'), pl.col('x2'))
                       - pl.max_horizontal(pl.col('x_min'), pl.col('x1')) > 0).alias('overlapping'))
        .filter(pl.col('overlapping'))
        .sort(by=["x_min", "x_max", pl.col("y1") + pl.col('y2')])
        .select(pl.col("x_min").alias('x1'),
                pl.col("x_max").alias("x2"),
                pl.col('y2').shift().over("x_min", 'x_max').alias('y1'),
                pl.col('y1').alias('y2')
                )
        .filter(pl.col('y2') - pl.col('y1') >= pct * (y_max - y_min))
        .sort(by=['y1', 'y2', 'x1'])
        .with_columns(((pl.col('x1') != pl.col('x2').shift())
                       | (pl.col('y1') != pl.col('y1').shift())
                       | (pl.col('y2') != pl.col('y2').shift())
                       ).cast(int).cum_sum().alias('ws_id')
                      )
        .group_by("ws_id")
        .agg(pl.col('x1').min().alias('x1'),
             pl.col('y1').min().alias('y1'),
             pl.col('x2').max().alias('x2'),
             pl.col('y2').max().alias('y2'))
        .drop("ws_id")
        .collect()
    )

    whitespaces = [Cell(**ws_dict) for ws_dict in df_elements_ranges.to_dicts()]

    # Flip object coordinates in horizontal case
    if not vertical:
        whitespaces = [Cell(x1=ws.y1, y1=ws.x1, x2=ws.y2, y2=ws.x2) for ws in whitespaces]

    return whitespaces


def adjacent_whitespaces(w_1: Cell, w_2: Cell) -> bool:
    """
    Identify if two whitespaces are adjacent
    :param w_1: first whitespace
    :param w_2: second whitespace
    :return: boolean indicating if two whitespaces are adjacent
    """
    x_coherent = len({w_1.x1, w_1.x2}.intersection({w_2.x1, w_2.x2})) > 0
    y_coherent = min(w_1.y2, w_2.y2) - max(w_1.y1, w_2.y1) > 0

    return x_coherent and y_coherent


def identify_coherent_v_whitespaces(v_whitespaces: List[Cell], char_length: float) -> List[Cell]:
    """
    From vertical whitespaces, identify the most relevant ones according to height, width and relative positions
    :param v_whitespaces: list of vertical whitespaces
    :param char_length: average character width in image
    :return: list of relevant vertical delimiters
    """
    # Create vertical delimiters groups
    v_groups = cluster_items(items=v_whitespaces,
                             clustering_func=adjacent_whitespaces)

    # Keep only delimiters that represent at least 75% of the height of their group
    v_delims = [d for gp in v_groups
                for d in [d for d in gp if d.height >= 0.75 * max([d.height for d in gp])]]

    # Group once again delimiters and keep only highest one in group
    v_delim_groups = cluster_items(items=v_delims,
                                   clustering_func=adjacent_whitespaces)

    # For each group, select a delimiter that has the largest height
    final_delims = list()
    for gp in v_delim_groups:
        if gp:
            # Get x center of group
            x_center = (min([d.x1 for d in gp]) + max([d.x2 for d in gp]))

            # Filter on tallest delimiters
            tallest_delimiters = [d for d in gp if d.height == max([d.height for d in gp])]

            # Add delimiter closest to the center of the group
            closest_del = sorted(tallest_delimiters, key=lambda d: abs(d.x1 + d.x2 - x_center)).pop(0)
            final_delims.append(closest_del)

    # Add all whitespaces of the largest height
    max_height_ws = [ws for ws in v_whitespaces if ws.height == max([w.height for w in v_whitespaces])]

    return list(set(final_delims + max_height_ws))


def get_relevant_vertical_whitespaces(segment: Union[ImageSegment, DelimiterGroup], char_length: float,
                                      pct: float = 0.25) -> List[Cell]:
    """
    Identify vertical whitespaces that can be column delimiters
    :param segment: image segment
    :param char_length: average character width in image
    :param pct: minimum percentage of the segment height for a vertical whitespace
    :return: list of vertical whitespaces that can be column delimiters
    """
    # Identify vertical whitespaces
    v_whitespaces = get_whitespaces(segment=segment,
                                    vertical=True,
                                    pct=pct,
                                    min_width=0.5 * char_length)

    # Identify relevant vertical whitespaces that can be column delimiters
    vertical_delims = identify_coherent_v_whitespaces(v_whitespaces=v_whitespaces,
                                                      char_length=char_length)

    return vertical_delims



def identify_column_groups(table_segment: TableSegment, char_length: float,
                           median_line_sep: float) -> Optional[DelimiterGroup]:
    """
    Identify list of vertical delimiters that can be table columns in a table segment
    :param table_segment: table segment
    :param char_length: average character width in image
    :param median_line_sep: median line separation
    :return: delimiter group that can correspond to columns
    """
    # Identify vertical whitespaces in the table segment
    vertical_ws, unused_ws = get_vertical_whitespaces(table_segment=table_segment)

    if len(vertical_ws) == 0 or len(table_segment.elements) == 0:
        return None

    # Create delimiter group from whitespace
    delimiter_group = get_column_whitespaces(vertical_ws=vertical_ws,
                                             unused_ws=unused_ws,
                                             table_segment=table_segment,
                                             char_length=char_length,
                                             median_line_sep=median_line_sep)

    if len(delimiter_group.delimiters) >= 4 and len(delimiter_group.elements) > 0:
        return delimiter_group
    else:
        return None



def get_coherent_ws_height(vertical_ws: List[Cell], unused_ws: List[Cell],
                           elements: List[Cell]) -> Tuple[List[Cell], List[Cell]]:
    """
    Get whitespaces with coherent height in relationship with elements
    :param vertical_ws: vertical whitespaces from segment
    :param unused_ws: list of unused whitespaces
    :param elements: elements in segment
    :return: tuple containing list of vertical whitespaces and list of unused whitespaces resized
    """
    # Define relevant ws
    relevant_ws = [ws for ws in unused_ws if ws.height >= 0.66 * max([w.height for w in vertical_ws])]
    relevant_ws += vertical_ws

    # Group elements in rows
    seq = iter(sorted(elements, key=lambda el: (el.y1, el.y2)))
    rows = [[next(seq)]]
    for el in seq:
        y2_row = max([el.y2 for el in rows[-1]])
        if el.y1 >= y2_row:
            rows.append([])
        rows[-1].append(el)
    
    # Identify top and bottom values for vertical whitespaces
    y_top, y_bottom, = max([ws.y2 for ws in relevant_ws]), min([ws.y1 for ws in relevant_ws])
    for row in rows:
        x1_row, x2_row = min([el.x1 for el in row]), max([el.x2 for el in row])
        y1_row, y2_row = min([el.y1 for el in row]), max([el.y2 for el in row])

        # Identify whitespaces that correspond vertically to rows
        row_ws = [ws for ws in relevant_ws
                  if min(ws.y2, y2_row) - max(ws.y1, y1_row) == y2_row - y1_row]

        if len([ws for ws in row_ws if min(ws.x2, x2_row) - max(ws.x1, x1_row) > 0]) > 0:
            y_top = min(y_top, y1_row)
            y_bottom = max(y_bottom, y2_row)

    # Reprocess whitespaces
    vertical_ws = [Cell(x1=ws.x1, y1=max(ws.y1, y_top), x2=ws.x2, y2=min(ws.y2, y_bottom)) for ws in vertical_ws]
    unused_ws = [Cell(x1=ws.x1, y1=max(ws.y1, y_top), x2=ws.x2, y2=min(ws.y2, y_bottom)) for ws in unused_ws]

    return vertical_ws, unused_ws


def corresponding_whitespaces(ws_1: Cell, ws_2: Cell, char_length: float, median_line_sep: float) -> bool:
    """
    Identify if whitespaces can correspond vertically
    :param ws_1: first whitespace
    :param ws_2: second whitespace
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: boolean indicating if whitespaces can correspond vertically
    """
    if min(abs(ws_2.y2 - ws_1.y1), abs(ws_1.y2 - ws_2.y1),
           abs(ws_1.y1 - ws_2.y1), abs(ws_2.y2 - ws_1.y2)) > 2 * median_line_sep:
        return False

    return min(ws_1.x2, ws_2.x2) - max(ws_1.x1, ws_2.x1) >= -char_length / 2


def identify_missing_vertical_whitespaces(unused_ws: List[Cell], char_length: float, median_line_sep: float,
                                          ref_height: int) -> List[Cell]:
    """
    Identify potential missing delimiters
    :param unused_ws: list of unused whitespace
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param ref_height: reference height
    :return: list of newly created whitespaces
    """
    # Create clusters of corresponding whitespaces
    f_cluster = partial(corresponding_whitespaces, char_length=char_length, median_line_sep=median_line_sep)
    ws_clusters = cluster_items(items=unused_ws,
                                clustering_func=f_cluster)

    new_ws = list()
    # Check if clusters can create a new vertical whitespace
    for cl in ws_clusters:
        if max([ws.y2 for ws in cl]) - min([ws.y1 for ws in cl]) >= 0.66 * ref_height:
            v_ws = Cell(x1=min([ws.x1 for ws in cl]),
                        y1=min([ws.y1 for ws in cl]),
                        x2=max([ws.x2 for ws in cl]),
                        y2=max([ws.y2 for ws in cl]))
            new_ws.append(v_ws)

    return new_ws


def distance_to_elements(x: int, elements: List[Cell]) -> Tuple[int, float]:
    """
    Compute distance metrics of elements to an x value
    :param x: x value
    :param elements: elements
    :return: distance / number of avoided elements
    """
    distance_elements = 0
    number_avoided_elements = 0
    for el in elements:
        if el.x1 <= x <= el.x2:
            distance_elements -= min(abs(el.x1 - x), abs(el.x2 - x)) ** 1 / 3
        else:
            number_avoided_elements += 1
            distance_elements += min(abs(el.x1 - x), abs(el.x2 - x)) ** 1 / 3

    return number_avoided_elements, distance_elements


def get_coherent_whitespace_position(ws: Cell, elements: List[Cell]) -> Cell:
    """
    Get coherent whitespace position for whitespace in relationship to segment elements
    :param ws: whitespace
    :param elements: segment elements
    :return: final whitespace
    """
    # Get potential conflicting elements
    conflicting_els = [el for el in elements if min(el.x2, ws.x2) - max(el.x1, ws.x1) > 0
                       and min(el.y2, ws.y2) - max(el.y1, ws.y1) > 0]

    if conflicting_els:
        # Get x value that maximises the distance to conflicting elements
        x_ws = sorted(range(ws.x1, ws.x2 + 1),
                      key=lambda x: distance_to_elements(x, conflicting_els),
                      reverse=True)[0]
    else:
        # Get elements to left and right of ws
        left_els = [el for el in elements if min(el.y2, ws.y2) - max(el.y1, ws.y1) > 0 and el.x2 <= ws.x1]
        right_els = [el for el in elements if min(el.y2, ws.y2) - max(el.y1, ws.y1) > 0 and ws.x2 <= el.x1]

        if len(left_els) > 0 and len(right_els) > 0:
            x_ws = round((max([el.x2 for el in left_els]) + min([el.x1 for el in right_els])) / 2)
        elif len(left_els) > 0:
            x_ws = max([el.x2 for el in left_els])
        elif len(right_els) > 0:
            x_ws = min([el.x1 for el in right_els])
        else:
            x_ws = round((ws.x1 + ws.x2) / 2)

    return Cell(x1=x_ws, y1=ws.y1, x2=x_ws, y2=ws.y2)


def filter_coherent_delimiters(delimiters: List[Cell], elements: List[Cell]) -> List[Cell]:
    # Check delimiters coherency (i.e) if it adds value
    filtered_delims = list()
    for delim in delimiters:
        left_delims = sorted([d for d in delimiters if d != delim and d.x2 < delim.x1
                              and min(delim.y2, d.y2) - max(delim.y1, d.y1) > 0
                              and d.height >= delim.height],
                             key=lambda d: d.x2)
        right_delims = sorted([d for d in delimiters if d != delim and d.x1 > delim.x2
                               and min(delim.y2, d.y2) - max(delim.y1, d.y1) > 0
                               and d.height >= delim.height],
                              key=lambda d: d.x1,
                              reverse=True)
        if len(right_delims) > 0 and len(left_delims) > 0:
            left_delim, right_delim = left_delims.pop(), right_delims.pop()
            # Get elements between delimiters
            left_els = [el for el in elements if el.x1 >= left_delim.x2 and el.x2 <= delim.x1
                        and min(el.y2, min(delim.y2, left_delim.y2)) - max(el.y1, max(delim.y1, left_delim.y1)) > 0]
            right_els = [el for el in elements if el.x1 >= delim.x2 and el.x2 <= right_delim.x1
                         and min(el.y2, min(delim.y2, right_delim.y2)) - max(el.y1, max(delim.y1, right_delim.y1)) > 0]
            if len(left_els) * len(right_els) > 0:
                filtered_delims.append(delim)
        elif len(right_delims) > 0:
            right_delim = right_delims.pop()
            # Get elements between delimiters
            right_els = [el for el in elements if el.x1 >= delim.x2 and el.x2 <= right_delim.x1
                         and min(el.y2, min(delim.y2, right_delim.y2)) - max(el.y1, max(delim.y1, right_delim.y1)) > 0]
            if len(right_els) > 0:
                filtered_delims.append(delim)
        elif len(left_delims) > 0:
            left_delim = left_delims.pop()
            # Get elements between delimiters
            left_els = [el for el in elements if el.x1 >= left_delim.x2 and el.x2 <= delim.x1
                        and min(el.y2, min(delim.y2, left_delim.y2)) - max(el.y1, max(delim.y1, left_delim.y1)) > 0]
            if len(left_els) > 0:
                filtered_delims.append(delim)
        else:
            filtered_delims.append(delim)

    return filtered_delims


def get_column_whitespaces(vertical_ws: List[Cell], unused_ws: List[Cell],
                           table_segment: TableSegment, char_length: float, median_line_sep: float) -> DelimiterGroup:
    """
    Identify all whitespaces that can be used as column delimiters in the table segment
    :param vertical_ws: list of vertical whitespaces in table segment
    :param unused_ws: list of unused whitespaces in table segment
    :param table_segment: table segment
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: delimiter group
    """
    # Get whitespaces with coherent height in relationship with elements
    vertical_ws, unused_ws = get_coherent_ws_height(vertical_ws=vertical_ws,
                                                    unused_ws=unused_ws,
                                                    elements=table_segment.elements)

    # Identify potential missing delimiters
    ref_height = max([ws.y2 for ws in vertical_ws]) - min([ws.y1 for ws in vertical_ws])
    missing_ws = identify_missing_vertical_whitespaces(unused_ws=unused_ws,
                                                       char_length=char_length,
                                                       median_line_sep=median_line_sep,
                                                       ref_height=ref_height)

    # Get final delimiters positions
    final_delims = list(set([get_coherent_whitespace_position(ws=ws,
                                                              elements=table_segment.elements)
                             for ws in vertical_ws + missing_ws]))

    # Filtered useful delimiters
    useful_delims = filter_coherent_delimiters(delimiters=final_delims,
                                               elements=table_segment.elements)

    # Create delimiter group
    x1_del, x2_del = min([d.x1 for d in useful_delims]), max([d.x2 for d in useful_delims])
    y1_del, y2_del = min([d.y1 for d in useful_delims]), max([d.y2 for d in useful_delims])
    delimiter_group = DelimiterGroup(delimiters=useful_delims,
                                     elements=[el for el in table_segment.elements if el.x1 >= x1_del
                                               and el.x2 <= x2_del and el.y1 >= y1_del and el.y2 <= y2_del])

    return delimiter_group




def deduplicate_whitespaces(vertical_whitespaces: List[VertWS], elements: List[Cell]) -> List[VertWS]:
    """
    Deduplicate adjacent vertical whitespaces
    :param vertical_whitespaces: list of VertWS objects
    :param elements: list of elements in segment
    :return: deduplicated vertical whitespaces
    """
    # Identify maximum height of whitespaces
    max_ws_height = max([ws.height for ws in vertical_whitespaces])

    # Create clusters of adjacent whitespaces
    ws_clusters = cluster_items(items=vertical_whitespaces,
                                clustering_func=adjacent_whitespaces)

    # For each group, get the tallest whitespace
    dedup_ws = list()
    for cl in ws_clusters:
        # Get x center of cluster
        x_center = min([ws.x1 for ws in cl]) + max([ws.x2 for ws in cl])

        # Filter on tallest delimiters
        max_cl_height = max([ws.height for ws in cl])
        tallest_ws = [ws for ws in cl if ws.height == max_cl_height]

        if max_cl_height == max_ws_height:
            dedup_ws += tallest_ws
        else:
            # Add whitespace closest to the center of the group
            closest_ws = sorted(tallest_ws, key=lambda ws: abs(ws.x1 + ws.x2 - x_center)).pop(0)
            dedup_ws.append(closest_ws)

    # Finally remove consecutive whitespaces that do not have elements between them
    dedup_ws = sorted(dedup_ws, key=lambda ws: ws.x1 + ws.x2)
    ws_to_del = list()
    for ws_left, ws_right in zip(dedup_ws, dedup_ws[1:]):
        # Get common area
        common_area = Cell(x1=ws_left.x2,
                           y1=max(ws_left.y1, ws_right.y1),
                           x2=ws_right.x1,
                           y2=min(ws_left.y2, ws_right.y2))

        # Identify matching elements
        matching_elements = [el for el in elements if el.x1 >= common_area.x1 and el.x2 <= common_area.x2
                             and el.y1 >= common_area.y1 and el.y2 <= common_area.y2]

        if len(matching_elements) == 0:
            # Add smallest element to deleted ws
            ws_to_del.append(ws_left if ws_left.height < ws_right.height else ws_right)

    return [ws for ws in dedup_ws if ws not in ws_to_del]


def get_vertical_whitespaces(table_segment: TableSegment) -> Tuple[List[Cell], List[Cell]]:
    """
    Identify vertical whitespaces as well as unused whitespaces in table segment
    :param table_segment: TableSegment object
    :return: tuple containing list of vertical whitespaces and list of unused whitespaces
    """
    # Identify all whitespaces x values
    x_ws = sorted(set([ws.x1 for ws in table_segment.whitespaces] + [ws.x2 for ws in table_segment.whitespaces]))

    # Get vertical whitespaces
    vertical_ws = list()
    for x_left, x_right in zip(x_ws, x_ws[1:]):
        # Create a whitespace object
        vert_ws = VertWS(x1=x_left, x2=x_right)

        for tb_area in table_segment.table_areas:
            # Get matching whitespaces
            matching_ws = [ws for ws in tb_area.whitespaces if min(vert_ws.x2, ws.x2) - max(vert_ws.x1, ws.x1) > 0]

            if matching_ws:
                vert_ws.add_position(tb_area.position)
                vert_ws.add_ws(matching_ws)

        # If it is composed of continuous whitespaces, use them
        if vert_ws.continuous:
            vertical_ws.append(vert_ws)

    # Filter whitespaces by height
    max_height = max([ws.height for ws in vertical_ws])
    vertical_ws = [ws for ws in vertical_ws if ws.height >= 0.66 * max_height]

    # Identify segment whitespaces that are unused
    unused_ws = [ws for ws in table_segment.whitespaces
                 if ws not in [ws for v_ws in vertical_ws for ws in v_ws.whitespaces]]

    # Deduplicate adjacent vertical delimiters
    vertical_ws = deduplicate_whitespaces(vertical_whitespaces=vertical_ws,
                                          elements=table_segment.elements)

    return [ws.cell for ws in vertical_ws], unused_ws



def segment_image(thresh: np.ndarray, lines: List[Line], char_length: float, median_line_sep: float) -> List[TableSegment]:
    """
    Segment image and its elements
    :param thresh: thresholded image array
    :param lines: list of Line objects of the image
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: list of ImageSegment objects with corresponding elements
    """
    # Identify image elements
    img_elements = get_image_elements(thresh=thresh,
                                      lines=lines,
                                      char_length=char_length,
                                      median_line_sep=median_line_sep)

    # Identify column segments
    y_min, y_max = min([el.y1 for el in img_elements]), max([el.y2 for el in img_elements])
    image_segment = ImageSegment(x1=0, y1=y_min, x2=thresh.shape[1], y2=y_max, elements=img_elements)

    col_segments = segment_image_columns(image_segment=image_segment,
                                         char_length=char_length,
                                         lines=lines)

    # Within each column, identify segments that can correspond to tables
    tb_segments = [table_segment for col_segment in col_segments
                   for table_segment in get_table_segments(segment=col_segment,
                                                           char_length=char_length,
                                                           median_line_sep=median_line_sep)
                   ]

    return tb_segments




def identify_remaining_segments(searched_rectangle: Rectangle,
                                existing_segments: List[Union[Cell, ImageSegment]]) -> List[Cell]:
    """
    Identify remaining segments in searched rectangle
    :param searched_rectangle: rectangle corresponding to area of research
    :param existing_segments: list of existing image segments
    :return: list of whitespaces as Cell objects
    """
    # Create rectangle objects from inputs
    obstacles = [Rectangle.from_cell(cell=element) for element in existing_segments]

    # Initiate queue
    queue = PriorityQueue()
    queue.put([-searched_rectangle.area, searched_rectangle, obstacles])

    segments = list()
    while not queue.qsize() == 0:
        q, r, obs = queue.get()
        if len(obs) == 0:
            # Update segments
            segments.append(r)

            # Update elements in queue
            for element in queue.queue:
                if element[1].overlaps(r):
                    element[2] += [r]

            continue

        # Get most pertinent obstacle
        pivot = sorted(obs, key=lambda o: o.distance(r))[0]

        # Create new rectangles
        rects = [Rectangle(x1=pivot.x2, y1=r.y1, x2=r.x2, y2=r.y2),
                 Rectangle(x1=r.x1, y1=r.y1, x2=pivot.x1, y2=r.y2),
                 Rectangle(x1=r.x1, y1=pivot.y2, x2=r.x2, y2=r.y2),
                 Rectangle(x1=r.x1, y1=r.y1, x2=r.x2, y2=pivot.y1)]

        for rect in rects:
            if rect.area > searched_rectangle.area / 100:
                rect_obstacles = [o for o in obs if o.overlaps(rect)]
                queue.put([-rect.area + random.uniform(0, 1), rect, rect_obstacles])

    return [seg.cell for seg in segments]


def get_vertical_ws(image_segment: ImageSegment, char_length: float, lines: List[Line]) -> List[Cell]:
    """
    Identify vertical whitespaces that can correspond to column delimiters in document
    :param image_segment: segment corresponding to the image
    :param char_length: average character length
    :param lines: list of lines identified in image
    :return: list of vertical whitespaces that can correspond to column delimiters in document
    """
    # Identify vertical whitespaces in segment that represent at least half of the image segment
    v_ws = get_whitespaces(segment=image_segment, vertical=True, pct=0.5)
    v_ws = [ws for ws in v_ws if ws.width >= char_length or ws.x1 == image_segment.x1 or ws.x2 == image_segment.x2]

    if len(v_ws) == 0:
        return []

    # Cut whitespaces with horizontal lines
    line_ws = list()
    h_lines = [l for l in lines if l.horizontal]
    for ws in v_ws:
        # Get crossing h_lines
        crossing_h_lines = sorted([l for l in h_lines if ws.y1 < l.y1 < ws.y2
                                   and min(ws.x2, l.x2) - max(ws.x1, l.x1) >= 0.5 * ws.width],
                                  key=lambda l: l.y1)
        if len(crossing_h_lines) > 0:
            # Get y values from whitespace and crossing lines
            y_values = sorted([ws.y1, ws.y2]
                              + [l.y1 - l.thickness for l in crossing_h_lines]
                              + [l.y1 + l.thickness for l in crossing_h_lines])

            # Create new sub whitespaces that are between two horizontal lines
            for y_top, y_bottom in [y_values[idx:idx + 2] for idx in range(0, len(y_values), 2)]:
                if y_bottom - y_top >= 0.5 * image_segment.height:
                    new_ws = Cell(x1=ws.x1, y1=y_top, x2=ws.x2, y2=y_bottom)
                    line_ws.append(new_ws)
        else:
            line_ws.append(ws)

    if len(line_ws) == 0:
        return []

    # Create groups of adjacent whitespaces
    line_ws = sorted(line_ws, key=lambda ws: ws.x1 + ws.x2)
    seq = iter(line_ws)

    line_ws_groups = [[next(seq)]]
    for ws in seq:
        prev_ws = line_ws_groups[-1][-1]

        # Get area delimited by the two whitespaces
        x1_area, x2_area = min(prev_ws.x2, ws.x1), max(prev_ws.x2, ws.x1)
        y1_area, y2_area = max(prev_ws.y1, ws.y1), min(prev_ws.y2, ws.y2)
        area = Cell(x1=x1_area, y1=y1_area, x2=x2_area, y2=y2_area)

        # Get separating elements
        separating_elements = [el for el in image_segment.elements if el.x1 >= area.x1 and el.x2 <= area.x2
                               and el.y1 >= area.y1 and el.y2 <= area.y2]

        if len(separating_elements) > 0:
            line_ws_groups.append([])
        line_ws_groups[-1].append(ws)

    # Keep only the tallest whitespace in each group
    final_ws = [sorted([ws for ws in cl if ws.height == max([w.height for w in cl])], key=lambda w: w.area).pop()
                for cl in line_ws_groups]

    return final_ws


def is_column_section(ws_group: List[Cell]) -> bool:
    """
    Identify if the whitespace group can correspond to columns
    :param ws_group: group of whitespaces
    :return: boolean indicating if the whitespace group can correspond to columns
    """
    # Check number of potential columns
    if not 3 <= len(ws_group) <= 4:
        return False

    # Check if column widths are consistent within the group
    ws_group = sorted(ws_group, key=lambda ws: ws.x1 + ws.x2)
    col_widths = [r_ws.x1 - l_ws.x2 for l_ws, r_ws in zip(ws_group, ws_group[1:])]

    return max(col_widths) / min(col_widths) <= 1.25


def identify_column_groups_image(image_segment: ImageSegment, vertical_ws: List[Cell]) -> List[List[Cell]]:
    """
    Identify groups of whitespaces that correspond to document columns
    :param image_segment: segment corresponding to the image
    :param vertical_ws: list of vertical whitespaces that can correspond to column delimiters in document
    :return: groups of whitespaces that correspond to document columns
    """
    # Identify whitespaces in the middle of the image as well as on edges
    middle_ws = [ws for ws in vertical_ws if
                 len({ws.x1, ws.x2}.intersection({image_segment.x1, image_segment.x2})) == 0]
    edge_ws = [ws for ws in vertical_ws if len({ws.x1, ws.x2}.intersection({image_segment.x1, image_segment.x2})) > 0]

    # Create groups of columns based on top/bottom alignment
    top_matches = lambda col_1, col_2: abs(col_1.y1 - col_2.y1) / max(col_1.height, col_2.height) <= 0.05
    bottom_matches = lambda col_1, col_2: abs(col_1.y2 - col_2.y2) / max(col_1.height, col_2.height) <= 0.05

    top_col_groups = [cl + edge_ws for cl in cluster_items(items=middle_ws, clustering_func=top_matches)]
    bottom_col_groups = [cl + edge_ws for cl in cluster_items(items=middle_ws, clustering_func=bottom_matches)]

    # Identify groups that correspond to columns
    col_groups = sorted([gp for gp in top_col_groups + bottom_col_groups if is_column_section(ws_group=gp)],
                        key=len,
                        reverse=True)

    # Get groups that contain all relevant whitespaces
    filtered_col_groups = list()
    for col_gp in col_groups:
        y_min, y_max = min([ws.y1 for ws in col_gp]), max([ws.y2 for ws in col_gp])
        matching_ws = [ws for ws in vertical_ws if min(ws.y2, y_max) - max(ws.y1, y_min) > 0.2 * ws.height
                       and len({ws.x1, ws.x2}.intersection({image_segment.x1, image_segment.x2})) == 0]
        if len(set(matching_ws).difference(set(col_gp))) == 0:
            filtered_col_groups.append(col_gp)

    if len(filtered_col_groups) == 0:
        return []

    # Deduplicate column groups
    seq = iter(filtered_col_groups)
    dedup_col_groups = [next(seq)]
    for col_gp in seq:
        if not any([set(col_gp).intersection(set(gp)) == set(col_gp) for gp in dedup_col_groups]):
            dedup_col_groups.append(col_gp)

    return dedup_col_groups


def get_column_group_segments(col_group: List[Cell]) -> List[ImageSegment]:
    """
    Identify image segments from the column group
    :param col_group: group of whitespaces that correspond to document columns
    :return: list of image segments defined by the column group
    """
    # Compute segments delimited by columns
    col_group = sorted(col_group, key=lambda ws: ws.x1 + ws.x2)
    col_segments = list()

    for left_ws, right_ws in zip(col_group, col_group[1:]):
        y1_segment, y2_segment = max(left_ws.y1, right_ws.y1), min(left_ws.y2, right_ws.y2)
        x1_segment, x2_segment = round((left_ws.x1 + left_ws.x2) / 2), round((right_ws.x1 + right_ws.x2) / 2)
        segment = ImageSegment(x1=x1_segment, y1=y1_segment, x2=x2_segment, y2=y2_segment)
        col_segments.append(segment)

    # Create rectangle defined by segments and identify remaining segments in area
    cols_rectangle = Rectangle(x1=min([seg.x1 for seg in col_segments]),
                               y1=min([seg.y1 for seg in col_segments]),
                               x2=max([seg.x2 for seg in col_segments]),
                               y2=max([seg.y2 for seg in col_segments]))
    remaining_segments = [ImageSegment(x1=area.x1, y1=area.y1, x2=area.x2, y2=area.y2)
                          for area in identify_remaining_segments(searched_rectangle=cols_rectangle,
                                                                  existing_segments=col_segments)
                          ]

    return col_segments + remaining_segments


def get_segments_from_columns(image_segment: ImageSegment, column_groups: List[List[Cell]]) -> List[ImageSegment]:
    """
    Identify all segments in image from columns
    :param image_segment: segment corresponding to the image
    :param column_groups: groups of whitespaces that correspond to document columns
    :return: list of segments in image from columns
    """
    # Identify image segments from column groups
    col_group_segments = [seg for col_gp in column_groups
                          for seg in get_column_group_segments(col_group=col_gp)]

    # Identify segments outside of columns
    top_segment = ImageSegment(x1=image_segment.x1,
                               y1=image_segment.y1,
                               x2=image_segment.x2,
                               y2=min([seg.y1 for seg in col_group_segments]))
    bottom_segment = ImageSegment(x1=image_segment.x1,
                                  y1=max([seg.y2 for seg in col_group_segments]),
                                  x2=image_segment.x2,
                                  y2=image_segment.y2)
    left_segment = ImageSegment(x1=image_segment.x1,
                                y1=min([seg.y1 for seg in col_group_segments]),
                                x2=min([seg.x1 for seg in col_group_segments]),
                                y2=max([seg.y2 for seg in col_group_segments]))
    right_segment = ImageSegment(x1=max([seg.x2 for seg in col_group_segments]),
                                 y1=min([seg.y1 for seg in col_group_segments]),
                                 x2=image_segment.x2,
                                 y2=max([seg.y2 for seg in col_group_segments]))

    # Create image segments and identify missing segments
    img_segments = col_group_segments + [top_segment, bottom_segment, left_segment, right_segment]
    missing_segments = [ImageSegment(x1=area.x1, y1=area.y1, x2=area.x2, y2=area.y2)
                        for area in identify_remaining_segments(searched_rectangle=Rectangle.from_cell(image_segment),
                                                                existing_segments=img_segments)
                        ]

    return img_segments + missing_segments


def segment_image_columns(image_segment: ImageSegment, char_length: float, lines: List[Line]) -> List[ImageSegment]:
    """
    Create image segments by identifying columns
    :param image_segment: segment corresponding to the image
    :param char_length: average character length
    :param lines: list of lines identified in image
    :return: list of segments corresponding to image
    """
    # Identify vertical whitespaces that can correspond to column delimiters in document
    vertical_ws = get_vertical_ws(image_segment=image_segment,
                                  char_length=char_length,
                                  lines=lines)

    # Identify column groups
    column_groups = identify_column_groups_image(image_segment=image_segment,
                                           vertical_ws=vertical_ws)

    if len(column_groups) == 0:
        return [image_segment]

    # Identify all segments in image from columns
    col_segments = get_segments_from_columns(image_segment=image_segment,
                                             column_groups=column_groups)

    # Populate elements in groups
    final_segments = list()
    for segment in col_segments:
        segment_elements = [el for el in image_segment.elements if el.x1 >= segment.x1 and el.x2 <= segment.x2
                            and el.y1 >= segment.y1 and el.y2 <= segment.y2]
        if segment_elements:
            segment.set_elements(elements=segment_elements)
            final_segments.append(segment)

    return final_segments


def get_image_elements(thresh: np.ndarray, lines: List[Line], char_length: float,
                       median_line_sep: float,) -> List[Cell]:
    """
    Identify image elements
    :param thresh: thresholded image array
    :param lines: list of image rows
    :param char_length: average character length
    :param median_line_sep: median line separation
    :return: list of image elements
    """
    # Mask rows
    for l in lines:
        if l.horizontal and l.length >= 3 * char_length:
            cv2.rectangle(thresh, (l.x1 - l.thickness, l.y1), (l.x2 + l.thickness, l.y2), (0, 0, 0), 3 * l.thickness)
        elif l.vertical and l.length >= 2 * char_length:
            cv2.rectangle(thresh, (l.x1, l.y1 - l.thickness), (l.x2, l.y2 + l.thickness), (0, 0, 0), 3 * l.thickness)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (max(int(char_length), 1), max(int(median_line_sep // 6), 1)))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Get list of contours
    elements = list()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        elements.append(Cell(x1=x, y1=y, x2=x + w, y2=y + h))

    # Filter elements that are too small
    elements = [el for el in elements if min(el.height, el.width) >= char_length]

    return elements


def get_table_areas(segment: ImageSegment, char_length: float, median_line_sep: float) -> List[ImageSegment]:
    """
    Identify relevant table areas in segment
    :param segment: ImageSegment object
    :param char_length: average character length in image
    :param median_line_sep: median line separation
    :return: list of table areas of segment
    """
    # Identify horizontal whitespaces in segment that represent at least half of median line separation
    h_ws = get_whitespaces(segment=segment, vertical=False, pct=1, min_width=0.5 * median_line_sep)
    h_ws = sorted(h_ws, key=lambda ws: ws.y1)

    # Handle case where no whitespaces have been found by creating "fake" ws at the top or bottom
    if len(h_ws) == 0:
        h_ws = [Cell(x1=min([el.x1 for el in segment.elements]),
                     x2=max([el.x2 for el in segment.elements]),
                     y1=segment.y1,
                     y2=segment.y1),
                Cell(x1=min([el.x1 for el in segment.elements]),
                     x2=max([el.x2 for el in segment.elements]),
                     y1=segment.y2,
                     y2=segment.y2)
                ]

    # Create whitespaces at the top or the bottom if they are missing
    if h_ws[0].y1 > segment.y1:
        up_ws = Cell(x1=min([ws.x1 for ws in h_ws]),
                     x2=max([ws.x2 for ws in h_ws]),
                     y1=segment.y1,
                     y2=segment.y1)
        h_ws.insert(0, up_ws)

    if h_ws[-1].y2 < segment.y2:
        down_ws = Cell(x1=min([ws.x1 for ws in h_ws]),
                       x2=max([ws.x2 for ws in h_ws]),
                       y1=segment.y2,
                       y2=segment.y2)
        h_ws.append(down_ws)

    # Check in areas between horizontal whitespaces in order to identify if they can correspond to tables
    table_areas = list()
    idx = 0

    for up, down in zip(h_ws, h_ws[1:]):
        idx += 1
        # Get the delimited area
        delimited_area = Cell(x1=max(min(up.x1, down.x1) - int(char_length), 0),
                              y1=up.y2,
                              x2=min(max(up.x2, down.x2) + int(char_length), segment.x2),
                              y2=down.y1)

        # Identify corresponding elements and create a corresponding segment
        area_elements = [el for el in segment.elements if el.x1 >= delimited_area.x1 and el.x2 <= delimited_area.x2
                         and el.y1 >= delimited_area.y1 and el.y2 <= delimited_area.y2]
        seg_area = ImageSegment(x1=delimited_area.x1,
                                x2=delimited_area.x2,
                                y1=delimited_area.y1,
                                y2=delimited_area.y2,
                                elements=area_elements,
                                position=idx)

        if area_elements:
            # Identify vertical whitespaces in the area
            v_ws = get_relevant_vertical_whitespaces(segment=seg_area, char_length=char_length, pct=0.5)

            # Identify number of whitespaces that are not on borders
            middle_ws = [ws for ws in v_ws if ws.x1 != seg_area.x1 and ws.x2 != seg_area.x2]

            # If there can be at least 3 columns in area, it is a possible table area
            if len(middle_ws) >= 1:
                # Add edges whitespaces
                left_ws = Cell(x1=seg_area.x1,
                               y1=seg_area.y1,
                               x2=min([el.x1 for el in seg_area.elements]),
                               y2=seg_area.y2)
                right_ws = Cell(x1=max([el.x2 for el in seg_area.elements]),
                                y1=seg_area.y1,
                                x2=seg_area.x2,
                                y2=seg_area.y2)
                v_ws = [ws for ws in v_ws
                        if not is_contained_cell(inner_cell=ws, outer_cell=left_ws, percentage=0.1)
                        and not is_contained_cell(inner_cell=ws, outer_cell=right_ws, percentage=0.1)
                        and len({ws.y1, ws.y2}.intersection({seg_area.y1, seg_area.y2})) > 0]

                seg_area.set_whitespaces(whitespaces=sorted(v_ws + [left_ws, right_ws], key=lambda ws: ws.x1 + ws.x2))
                table_areas.append(seg_area)

    return table_areas


def merge_consecutive_ws(whitespaces: List[Cell]) -> List[Cell]:
    """
    Merge consecutive whitespaces
    :param whitespaces: list of original whitespaces
    :return: list of merged whitespaces
    """
    whitespaces = sorted(whitespaces, key=lambda ws: ws.x1 + ws.x2)

    seq = iter(whitespaces)
    ws_groups = [[next(seq)]]
    for ws in seq:
        if ws.x1 > ws_groups[-1][-1].x2:
            ws_groups.append([])
        ws_groups[-1].append(ws)

    return [Cell(x1=ws_gp[0].x1, x2=ws_gp[-1].x2,
                 y1=min([ws.y1 for ws in ws_gp]),
                 y2=max([ws.y2 for ws in ws_gp]))
            for ws_gp in ws_groups]


def coherent_table_areas(tb_area_1: ImageSegment, tb_area_2: ImageSegment, char_length: float, median_line_sep: float) -> bool:
    """
    Identify if two table areas are coherent
    :param tb_area_1: first table area
    :param tb_area_2: second table area
    :param char_length: average character length in image
    :param median_line_sep: median line separation
    :return: boolean indicating if the two table areas are coherent
    """
    # Compute vertical difference
    v_diff = min(abs(tb_area_1.y2 - tb_area_2.y1), abs(tb_area_2.y2 - tb_area_1.y1))

    # If areas are not consecutive or with too much separation, not coherent
    if abs(tb_area_1.position - tb_area_2.position) != 1 or v_diff > 2 * median_line_sep:
        return False

    # Get relevant whitespaces
    if tb_area_1.position < tb_area_2.position:
        ws_tb_1 = merge_consecutive_ws([ws for ws in tb_area_1.whitespaces if ws.y2 == tb_area_1.y2])
        ws_tb_2 = merge_consecutive_ws([ws for ws in tb_area_2.whitespaces if ws.y1 == tb_area_2.y1])
    else:
        ws_tb_1 = merge_consecutive_ws([ws for ws in tb_area_1.whitespaces if ws.y1 == tb_area_1.y1])
        ws_tb_2 = merge_consecutive_ws([ws for ws in tb_area_2.whitespaces if ws.y2 == tb_area_2.y2])

    if max(len(ws_tb_1), len(ws_tb_2)) < 4:
        return False

    # Check whitespaces coherency
    if len(ws_tb_1) >= len(ws_tb_2):
        dict_ws_coherency = {
            idx_1: [ws_2 for ws_2 in ws_tb_2
                    if min(ws_1.x2, ws_2.x2) - max(ws_1.x1, ws_2.x1) >= 0.5 * char_length]
            for idx_1, ws_1 in enumerate(ws_tb_1)
        }
    else:
        dict_ws_coherency = {
            idx_2: [ws_1 for ws_1 in ws_tb_1
                    if min(ws_1.x2, ws_2.x2) - max(ws_1.x1, ws_2.x1) >= 0.5 * char_length]
            for idx_2, ws_2 in enumerate(ws_tb_2) if ws_2.width
        }

    # Compute threshold for coherency
    threshold = 1 if min(len(ws_tb_1), len(ws_tb_2)) < 4 else 0.8

    return np.mean([int(len(v) == 1) for v in dict_ws_coherency.values()]) >= threshold


def table_segment_from_group(table_segment_group: List[ImageSegment]) -> ImageSegment:
    """
    Create table segment from group of corresponding ImageSegment objects
    :param table_segment_group: list of ImageSegment objects
    :return: ImageSegment corresponding to table
    """
    # Retrieve all elements
    elements = [el for seg in table_segment_group for el in seg.elements]
    whitespaces = [ws for seg in table_segment_group for ws in seg.whitespaces]

    # Create ImageSegment object
    table_segment = ImageSegment(x1=min([seg.x1 for seg in table_segment_group]),
                                 y1=min([seg.y1 for seg in table_segment_group]),
                                 x2=max([seg.x2 for seg in table_segment_group]),
                                 y2=max([seg.y2 for seg in table_segment_group]),
                                 elements=elements,
                                 whitespaces=whitespaces)
    
    return table_segment


def get_table_segments(segment: ImageSegment, char_length: float, median_line_sep: float) -> List[TableSegment]:
    """
    Identify relevant table areas in segment
    :param segment: ImageSegment object
    :param char_length: average character length in image
    :param median_line_sep: median line separation
    :return: list of image segments corresponding to tables
    """
    # Get table areas
    table_areas = get_table_areas(segment=segment, char_length=char_length, median_line_sep=median_line_sep)

    if len(table_areas) == 0:
        return []

    # Create groups of table areas
    table_areas = sorted(table_areas, key=lambda tb: tb.position)
    seq = iter(table_areas)
    tb_areas_gps = [[next(seq)]]
    for tb_area in seq:
        prev_table = tb_areas_gps[-1][-1]
        if not coherent_table_areas(tb_area_1=prev_table,
                                    tb_area_2=tb_area,
                                    char_length=char_length,
                                    median_line_sep=median_line_sep):
            tb_areas_gps.append([])
        tb_areas_gps[-1].append(tb_area)
        
    # Create image segments corresponding to potential table
    table_segments = [TableSegment(table_areas=tb_area_gp) for tb_area_gp in tb_areas_gps
                      if max([len(tb_area.whitespaces) for tb_area in tb_area_gp]) > 3]

    return table_segments



def detect_delimiter_group_rows(delimiter_group: DelimiterGroup) -> List[TableRow]:
    """
    Identify list of rows corresponding to the delimiter group
    :param delimiter_group: column delimiters group
    :return: list of rows corresponding to the delimiter group
    """
    # Identify list of rows corresponding to the delimiter group
    table_rows, median_row_sep = identify_delimiter_group_rows(delimiter_group=delimiter_group)

    return table_rows



def get_delimiter_group_row_separation(delimiter_group: DelimiterGroup) -> Optional[float]:
    """
    Identify median row separation between elements of the delimiter group
    :param delimiter_group: column delimiters group
    :return: median row separation in pixels
    """
    if len(delimiter_group.elements) == 0:
        return None

    # Create dataframe with delimiter group elements
    list_elements = [{"id": idx, "x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2}
                     for idx, el in enumerate(delimiter_group.elements)]
    df_elements = pl.LazyFrame(data=list_elements)

    # Cross join to get corresponding elements and filter on elements that corresponds horizontally
    df_h_elms = (df_elements.join(df_elements, how='cross')
                 .filter(pl.col('id') != pl.col('id_right'))
                 .filter(pl.min_horizontal(['x2', 'x2_right']) - pl.max_horizontal(['x1', 'x1_right']) > 0)
                 )

    # Get element which is directly below
    df_elms_below = (df_h_elms.filter(pl.col('y1') < pl.col('y1_right'))
                     .sort(['id', 'y1_right'])
                     .with_columns(pl.lit(1).alias('ones'))
                     .with_columns(pl.col('ones').cum_sum().over(["id"]).alias('rk'))
                     .filter(pl.col('rk') == 1)
                     )

    if df_elms_below.collect().height == 0:
        return None

    # Compute median vertical distance between elements
    median_v_dist = (df_elms_below.with_columns(((pl.col('y1_right') + pl.col('y2_right')
                                                  - pl.col('y1') - pl.col('y2')) / 2).abs().alias('y_diff'))
                     .select(pl.median('y_diff'))
                     .collect()
                     .to_dicts()
                     .pop()
                     .get('y_diff')
                     )

    return median_v_dist


def identify_aligned_elements(df_elements: pl.DataFrame, ref_size: int) -> List[Set[int]]:
    """
    Identify groups of elements that are vertically aligned
    :param df_elements: dataframe of elements
    :param ref_size: reference distance between two line centers
    :return: list of element sets which represent aligned elements
    """
    # Cross join elements and filter on aligned elements
    df_cross = (df_elements.join(df_elements, how="cross")
                .filter(((pl.col('y1') + pl.col('y2') - pl.col('y1_right') - pl.col('y2_right')) / 2).abs() <= ref_size)
                .filter(pl.max_horizontal(pl.col('height'), pl.col('height_right'))
                        / pl.min_horizontal(pl.col('height'), pl.col('height_right')) <= 2)
                .select(pl.struct(pl.min_horizontal(pl.col('idx'), pl.col('idx_right')).alias('idx_1'),
                                  pl.max_horizontal(pl.col('idx'), pl.col('idx_right')).alias('idx_2')).alias('idxs')
                        )
                .unique()
                .unnest("idxs")
                )
    aligned_elements = [{row.get('idx_1'), row.get('idx_2')} for row in df_cross.to_dicts()]

    # Create cluster of aligned elements
    clusters = find_components(edges=aligned_elements)

    return clusters


def identify_overlapping_row_clusters(df_elements: pl.DataFrame, clusters: List[Set[int]]) -> List[Set[int]]:
    """
    Identify rows that overlap with each other and group them in cluster
    :param df_elements: dataframe of elements
    :param clusters: list of element sets which represent aligned elements/ rows
    :return: list of clusters sets which represent aligned rows
    """
    clusters_dict = {el_idx: cl_idx for cl_idx, cl in enumerate(clusters) for el_idx in cl}
    df_elements = df_elements.with_columns(pl.col('idx').map_elements(clusters_dict.get).alias("cl_idx"))

    df_clusters = (df_elements.group_by("cl_idx")
                   .agg(pl.col("x1").min().alias('x1'),
                        pl.col("y1").min().alias('y1'),
                        pl.col("x2").max().alias('x2'),
                        pl.col("y2").max().alias('y2'))
                   .with_columns((pl.col("y2") - pl.col("y1")).alias('height'))
                   )

    df_cross = (df_clusters.join(df_clusters, how='cross')
                .with_columns((pl.min_horizontal(pl.col("y2"), pl.col("y2_right"))
                               - pl.max_horizontal(pl.col("y1"), pl.col("y1_right"))).alias('overlap')
                             )
                .filter(pl.col('overlap') / pl.min_horizontal(pl.col("height"), pl.col("height_right")) >= 0.5)
                .select(pl.struct(pl.min_horizontal(pl.col('cl_idx'), pl.col('cl_idx_right')).alias('idx_1'),
                                  pl.max_horizontal(pl.col('cl_idx'), pl.col('cl_idx_right')).alias('idx_2')).alias('idxs')
                        )
                .unique()
                .unnest("idxs")
                )
    aligned_rows = [{row.get('idx_1'), row.get('idx_2')} for row in df_cross.to_dicts()]

    # Create cluster of aligned rows
    row_clusters = find_components(edges=aligned_rows)

    return row_clusters


def not_overlapping_rows(tb_row_1: TableRow, tb_row_2: TableRow) -> bool:
    """
    Identify if two TableRow objects do not overlap vertically
    :param tb_row_1: first TableRow object
    :param tb_row_2: second TableRow object
    :return: boolean indicating if both TableRow objects do not overlap vertically
    """
    # Compute overlap
    overlap = min(tb_row_1.y2, tb_row_2.y2) - max(tb_row_1.y1, tb_row_2.y1)
    return overlap / min(tb_row_1.height, tb_row_2.height) <= 0.1


def score_row_group(row_group: List[TableRow], height: int, max_elements: int) -> float:
    """
    Score row group pertinence
    :param row_group: group of TableRow objects
    :param height: reference height
    :param max_elements: reference number of elements/cells that can be included in a row group
    :return: scoring of the row group
    """
    # Get y coverage of row group
    y_total = sum([r.height for r in row_group])
    y_overlap = sum([max(0, min(r_1.y2, r_2.y2) - max(r_1.y1, r_2.y1)) for r_1, r_2 in zip(row_group, row_group[1:])])
    y_coverage = y_total - y_overlap

    # Score row group
    return (sum([len(r.cells) for r in row_group]) / max_elements) * (y_coverage / height)


def get_rows_from_overlapping_cluster(row_cluster: List[TableRow]) -> List[TableRow]:
    """
    Identify relevant rows from a cluster of vertically overlapping rows
    :param row_cluster: cluster of vertically overlapping TableRow objects
    :return: relevant rows from a cluster of vertically overlapping rows
    """
    # Get height of row cluster
    ref_height = max([r.y2 for r in row_cluster]) - min([r.y1 for r in row_cluster])

    # Get groups of distinct rows
    seq = iter(row_cluster)
    distinct_rows_clusters = [[next(seq)]]
    for row in seq:
        for idx, cl in enumerate(distinct_rows_clusters):
            if all([not_overlapping_rows(tb_row_1=row, tb_row_2=r) for r in cl]):
                distinct_rows_clusters[idx].append(row)
        distinct_rows_clusters.append([row])

    # Get maximum number of elements possible in a row cluster
    max_elements = max([sum([len(r.cells) for r in cl]) for cl in distinct_rows_clusters])

    # Sort elements by score
    scored_elements = sorted(distinct_rows_clusters,
                             key=lambda gp: score_row_group(row_group=gp,
                                                            height=ref_height,
                                                            max_elements=max_elements)
                             )

    # Get cluster of rows with the largest score
    return scored_elements.pop()


def identify_rows(elements: List[Cell], ref_size: int) -> List[TableRow]:
    """
    Identify rows from Cell elements
    :param elements: list of cells
    :param ref_size: reference distance between two line centers
    :return: list of table rows
    """
    if len(elements) == 0:
        return []

    # Create dataframe with elements
    df_elements = (pl.DataFrame([{"idx": idx, "x1": el.x1, "y1": el.y1, "x2": el.x2, "y2": el.y2}
                                 for idx, el in enumerate(elements)])
                   .with_columns((pl.col('y2') - pl.col('y1')).alias('height'))
                   )

    # Identify clusters of elements that represent rows
    element_clusters = identify_aligned_elements(df_elements=df_elements,
                                                 ref_size=ref_size)

    # Identify overlapping rows
    row_clusters = identify_overlapping_row_clusters(df_elements=df_elements,
                                                     clusters=element_clusters)

    overlap_row_clusters = [[TableRow(cells=[elements[idx] for idx in element_clusters[cl_idx]]) for cl_idx in row_cl]
                            for row_cl in row_clusters]

    # Get relevant rows in each cluster
    relevant_rows = [row for cl in overlap_row_clusters
                     for row in get_rows_from_overlapping_cluster(cl)]

    # Check for overlapping rows
    seq = iter(sorted(relevant_rows, key=lambda r: r.y1 + r.y2))
    final_rows = [next(seq)]
    for row in seq:
        if row.overlaps(final_rows[-1]):
            final_rows[-1].merge(row)
        else:
            final_rows.append(row)

    return final_rows


def identify_delimiter_group_rows(delimiter_group: DelimiterGroup) -> Tuple[List[TableRow], float]:
    """
    Identify list of rows corresponding to the delimiter group
    :param delimiter_group: column delimiters group
    :return: list of rows corresponding to the delimiter group
    """
    # Identify median row separation between elements of the delimiter group
    group_median_row_sep = get_delimiter_group_row_separation(delimiter_group=delimiter_group)

    if group_median_row_sep:
        # Identify rows
        group_lines = identify_rows(elements=delimiter_group.elements,
                                    ref_size=int(group_median_row_sep // 3))

        # Adjust height of first / last row
        if group_lines:
            group_lines[0].set_y_top(delimiter_group.y1)
            group_lines[-1].set_y_bottom(delimiter_group.y2)

        return group_lines, group_median_row_sep

    return [], group_median_row_sep



def identify_table(columns: DelimiterGroup, table_rows: List[TableRow], contours: List[Cell], median_line_sep: float,
                   char_length: float) -> Optional[Table]:
    """
    Identify table from column delimiters and rows
    :param columns: column delimiters group
    :param table_rows: list of table rows corresponding to columns
    :param contours: list of image contours
    :param median_line_sep: median line separation
    :param char_length: average character length
    :return: Table object
    """
    # Create table from rows and columns delimiters
    table = get_table(columns=columns,
                      table_rows=table_rows,
                      contours=contours)

    if table:
        if check_table_coherency(table=table,
                                 median_line_sep=median_line_sep,
                                 char_length=char_length):
            return table

    return None



def check_row_coherency(table: Table, median_line_sep: float) -> bool:
    """
    Check row coherency of table
    :param table: Table object
    :param median_line_sep: median line separation
    :return: boolean indicating if table row heights are coherent
    """
    if table.nb_rows < 2:
        return False

    # Get median row separation
    median_row_separation = np.median([(lower_row.y1 + lower_row.y2 - upper_row.y1 - upper_row.y2) / 2
                                       for upper_row, lower_row in zip(table.items, table.items[1:])])

    return median_row_separation >= median_line_sep / 3


def check_column_coherency(table: Table, char_length: float) -> bool:
    """
    Check column coherency of table
    :param table: Table object
    :param char_length: average character length
    :return: boolean indicating if table column widths are coherent
    """
    if table.nb_columns < 2:
        return False

    # Get column widths
    col_widths = list()
    for idx in range(table.nb_columns):
        col_elements = [row.items[idx] for row in table.items]
        col_width = min([el.x2 for el in col_elements]) - max([el.x1 for el in col_elements])
        col_widths.append(col_width)

    return np.median(col_widths) >= 3 * char_length


def check_table_coherency(table: Table, median_line_sep: float, char_length: float) -> bool:
    """
    Check if table has coherent dimensions
    :param table: Table object
    :param median_line_sep: median line separation
    :param char_length: average character length
    :return: boolean indicating if table dimensions are coherent
    """
    # Check row coherency of table
    row_coherency = check_row_coherency(table=table,
                                        median_line_sep=median_line_sep)

    # Check column coherency of table
    column_coherency = check_column_coherency(table=table,
                                              char_length=char_length)

    return row_coherency and column_coherency



def get_coherent_columns_dimensions(columns: DelimiterGroup, table_rows: List[TableRow]) -> DelimiterGroup:
    """
    Identify columns that encapsulate at least one row
    :param columns: column delimiters group
    :param table_rows: list of table rows
    :return: relevant columns according to table rows
    """
    original_delimiters = sorted(columns.delimiters, key=lambda delim: delim.x1 + delim.x2)

    # Get horizontal dimensions of rows
    x_min, x_max = min([row.x1 for row in table_rows]), max([row.x2 for row in table_rows])

    # Identify left and right delimiters
    left_delim = [delim for delim in original_delimiters if delim.x2 <= x_min][-1]
    right_delim = [delim for delim in original_delimiters if delim.x1 >= x_max][0]

    # Identify middle delimiters
    middle_delimiters = [delim for delim in original_delimiters if delim.x1 >= x_min and delim.x2 <= x_max]

    # Create new delimiter group
    delim_group = DelimiterGroup(delimiters=[left_delim] + middle_delimiters + [right_delim])

    return delim_group


def get_table(columns: DelimiterGroup, table_rows: List[TableRow], contours: List[Cell]) -> Table:
    """
    Create table object from column delimiters and rows
    :param columns: column delimiters group
    :param table_rows: list of table rows
    :param contours: list of image contours
    :return: Table object
    """
    # Identify coherent column delimiters in relationship to table rows
    coherent_columns = get_coherent_columns_dimensions(columns=columns,
                                                       table_rows=table_rows)

    # Compute vertical delimiters from rows
    lines = sorted(table_rows, key=lambda l: l.v_center)
    y_min = min([line.y1 for line in lines])
    y_max = max([line.y2 for line in lines])
    v_delims = [y_min] + [int(round((up.y2 + down.y1) / 2)) for up, down in zip(lines, lines[1:])] + [y_max]

    # Create cells for table
    list_cells = list()
    for y_top, y_bottom in zip(v_delims, v_delims[1:]):
        # Identify delimiters that correspond vertically to rows
        line_delims = [d for d in coherent_columns.delimiters if
                       min(d.y2, y_bottom) - max(d.y1, y_top) > (y_bottom - y_top) // 2]

        # Sort line delimiters and compute horizontal delimiters
        line_delims = sorted(line_delims, key=lambda d: d.x1)
        h_delims = [line_delims[0].x2] + [(d.x1 + d.x2) // 2 for d in line_delims[1:-1]] + [line_delims[-1].x1]

        for x_left, x_right in zip(h_delims, h_delims[1:]):
            cell = Cell(x1=x_left,
                        y1=y_top,
                        x2=x_right,
                        y2=y_bottom)
            list_cells.append(cell)

    # Create table object
    table = cluster_to_table(cluster_cells=list_cells, elements=contours, borderless=True)

    return table if table.nb_columns >= 3 and table.nb_rows >= 2 else None









