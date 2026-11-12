"""Lossless compact geometry for Qwen; persisted layout artifacts stay unchanged."""
import json


BLOCK_FIELDS = ["reading_order", "text", "x1_index", "y1_index", "x2_index", "y2_index", "confidence"]


def compact_layout_text(layout_text):
    """Replace repeated field names and coordinates with indexed arrays, without rounding.

    Only the known layout shape is encoded. Unknown shapes/extra fields are kept
    verbatim so this optimization cannot silently discard metadata from callers.
    """
    try:
        layout = json.loads(layout_text)
        if (not isinstance(layout, dict) or set(layout) != {"coordinate_system", "pages"}
                or not isinstance(layout["pages"], list)):
            return layout_text
        pages = []
        for page in layout["pages"]:
            if (not isinstance(page, dict) or not isinstance(page.get("blocks"), list)
                    or "x_coordinates" in page or "y_coordinates" in page):
                return layout_text
            xs, ys, x_indices, y_indices = [], [], {}, {}

            def index(value, coordinates, indices):
                # Use JSON spelling as the key to retain int/float representations.
                key = json.dumps(value, allow_nan=False)
                if key not in indices:
                    indices[key] = len(coordinates)
                    coordinates.append(value)
                return indices[key]

            rows = []
            for block in page["blocks"]:
                if (not isinstance(block, dict)
                        or set(block) != {"reading_order", "text", "bbox", "confidence"}
                        or not isinstance(block["bbox"], list) or len(block["bbox"]) != 4):
                    return layout_text
                x1, y1, x2, y2 = block["bbox"]
                rows.append([block["reading_order"], block["text"],
                             index(x1, xs, x_indices), index(y1, ys, y_indices),
                             index(x2, xs, x_indices), index(y2, ys, y_indices), block["confidence"]])
            pages.append({**{key: value for key, value in page.items() if key != "blocks"},
                          "x_coordinates": xs, "y_coordinates": ys, "blocks": rows})
        compact = json.dumps({"encoding": "indexed_axes_v1", "block_fields": BLOCK_FIELDS,
                              "coordinate_system": layout["coordinate_system"], "pages": pages},
                             ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return compact if len(compact) < len(layout_text) else layout_text
    except (ValueError, TypeError):
        return layout_text
