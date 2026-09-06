"""Tile evenly spaced frames of a film into one contact sheet.

This used to be an ffmpeg one-liner that read the frame count out of ffmpeg's
own progress output and turned it into a `select='not(mod(n,STRIDE))'` filter.
The parse came back empty once, the stride became 1, and the sheet showed the
first twelve frames of a 141-second film -- twelve near-identical pictures of
two ducks standing in a field. Counting frames is the reader's job, so it is
done here where a miscount is an error rather than a silently useless sheet.

    python -m microduck_lab.tools.contact_sheet FILM.mp4 [-o SHEET.jpg]
                                                [--cols 4] [--rows 3] [--width 400]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def frames(path, count):
    """`count` frames spread evenly across the film, first to last."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        total = reader.count_frames()
    except Exception:                       # some containers cannot be counted
        total = sum(1 for _ in reader)
        reader = imageio.get_reader(path)
    if total < count:
        raise SystemExit(f"{path}: {total} frames, fewer than the {count} asked for")
    picks = np.linspace(0, total - 1, count).round().astype(int)
    out = [np.asarray(reader.get_data(int(i))) for i in picks]
    reader.close()
    return out, total


def sheet(images, cols, width):
    import PIL.Image

    thumbs = []
    for im in images:
        img = PIL.Image.fromarray(im)
        img = img.resize((width, round(width * img.height / img.width)), PIL.Image.LANCZOS)
        thumbs.append(np.asarray(img))
    h, w = thumbs[0].shape[:2]
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for k, t in enumerate(thumbs):
        r, c = divmod(k, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = t[:h, :w, :3]
    return grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film")
    ap.add_argument("-o", "--out")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--width", type=int, default=400, help="thumbnail width in pixels")
    args = ap.parse_args()

    out = args.out or str(Path(args.film).with_name(Path(args.film).stem + "_sheet.jpg"))
    picked, total = frames(args.film, args.cols * args.rows)

    import imageio.v2 as imageio
    imageio.imwrite(out, sheet(picked, args.cols, args.width), quality=88)
    print(f"{out}: {args.cols}x{args.rows} frames spread over {total} ({args.film})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
