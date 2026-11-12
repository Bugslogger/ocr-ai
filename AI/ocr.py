"""Convert PaddleOCR detection/recognition results into invoice text rows."""
import math
from collections.abc import Mapping


def text_blocks(result):
    """Preserve recognition-array order and page-local geometry without sorting."""
    if not isinstance(result, Mapping):
        raise RuntimeError("Unsupported PaddleOCR result: expected a mapping")
    texts = result.get("rec_texts")
    boxes = result.get("rec_boxes")
    if texts is None or boxes is None or len(texts) != len(boxes):
        raise RuntimeError("PaddleOCR result must contain matching rec_texts and rec_boxes")
    scores = result.get("rec_scores")
    if scores is not None and len(scores) != len(texts):
        raise RuntimeError("PaddleOCR rec_scores must match rec_texts")
    blocks = []
    for index, (text, box) in enumerate(zip(texts, boxes)):
        if not isinstance(text, str) or len(box) != 4:
            raise RuntimeError("Malformed PaddleOCR text or bounding box")
        x1, y1, x2, y2 = map(float, box)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise RuntimeError("Invalid PaddleOCR bounding box")
        score = float(scores[index]) if scores is not None else None
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise RuntimeError("Invalid PaddleOCR recognition confidence")
        blocks.append({"reading_order": index, "text": text,
                       "bbox": [x1, y1, x2, y2], "confidence": score})
    return blocks


def page_text(result):
    """Group boxes geometrically into rows; retain tabs between columns."""
    cells = [(*block["bbox"], block["text"].strip()) for block in text_blocks(result)
             if block["text"].strip()]
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
