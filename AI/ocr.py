"""Convert PaddleOCR detection/recognition results into invoice text rows."""
import math
from collections.abc import Mapping


def page_text(result):
    """Group axis-aligned word boxes by row, then order cells left to right.

    This is geometric text ordering, not a table structure recognition model.
    Keep tabs between cells so Qwen can interpret invoice columns.
    """
    if not isinstance(result, Mapping):
        raise RuntimeError("Unsupported PaddleOCR result: expected a mapping")
    texts = result.get("rec_texts")
    boxes = result.get("rec_boxes")
    if texts is None or boxes is None or len(texts) != len(boxes):
        raise RuntimeError("PaddleOCR result must contain matching rec_texts and rec_boxes")
    cells = []
    for text, box in zip(texts, boxes):
        if not isinstance(text, str) or len(box) != 4:
            raise RuntimeError("Malformed PaddleOCR text or bounding box")
        x1, y1, x2, y2 = map(float, box)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise RuntimeError("Invalid PaddleOCR bounding box")
        if text.strip():
            cells.append((x1, y1, x2, y2, text.strip()))
    rows = []
    for cell in sorted(cells, key=lambda c: ((c[1] + c[3]) / 2, c[0])):
        center = (cell[1] + cell[3]) / 2
        height = cell[3] - cell[1]
        # Compare against the first cell so a tall box cannot merge many rows.
        row = next((row for row in reversed(rows)
                    if abs(center - row[0]) <= 0.5 * min(height, row[1])), None)
        if row is None:
            rows.append((center, height, [cell]))
        else:
            row[2].append(cell)
    return "\n".join("\t".join(c[4] for c in sorted(row[2], key=lambda c: c[0])) for row in rows)
