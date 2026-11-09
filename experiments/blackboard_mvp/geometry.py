"""Shared, pure rectangle geometry. XYWH ranges may extend outside the frame."""

def intersect(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    return [x, y, right-x, bottom-y] if right > x and bottom > y else None


def subtract(rect, cover):
    overlap = intersect(rect, cover)
    if overlap is None:
        return [list(rect)]
    x, y, w, h = rect
    ox, oy, ow, oh = overlap
    pieces = [[x, y, w, oy-y], [x, oy+oh, w, y+h-oy-oh],
              [x, oy, ox-x, oh], [ox+ow, oy, x+w-ox-ow, oh]]
    return [r for r in pieces if r[2] > 0 and r[3] > 0]


