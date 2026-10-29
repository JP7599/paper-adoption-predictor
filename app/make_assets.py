"""Draw the app's favicon: a dark green square with a thin hexagon. Run once:  .venv/bin/python app/make_assets.py"""
import math
from pathlib import Path

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)


def hexagon(cx, cy, r, rot):
    return [(cx + r * math.cos(math.radians(rot + 60 * i)), cy + r * math.sin(math.radians(rot + 60 * i))) for i in range(6)]


def favicon(name="favicon.png", size=64):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, size); ax.set_ylim(0, size); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), size, size, color="#1f4d3a"))
    ax.add_patch(plt.Polygon(hexagon(size / 2, size / 2, 21, 30), fill=False, ec="#ffffff", lw=3.2, joinstyle="round"))
    ax.plot([size / 2], [size / 2], "o", color="#ffffff", ms=5)
    fig.savefig(OUT / name, dpi=100); plt.close(fig)


if __name__ == "__main__":
    favicon()
    print("wrote", OUT / "favicon.png")
