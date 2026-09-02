# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Figures 1 and 3 of article 1: the geometry of the mask classes.

Figure 1 is one bin of the complex plane. A real mask can only return a point of
the line R.x, the best such point is the orthogonal projection of s onto it, and
the residual is the orthogonal complement, of length |s| sin(theta).

Figure 3 is the ladder of 2x2 block structures, drawn as what each does to the
unit circle: homothety, similitude, general linear map, and the affine variant
that moves the image off the origin. The matrices here are synthetic, chosen to
be readable; the paper's version reads them off a fitted model.

    py -3.14 -m uv run figures/geometry.py [outdir]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

X = np.array([1.5, 0.35])  # mixture, as a vector of R^2
S = np.array([0.95, 1.05])  # source


def project(s: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Proposition 1 in two dimensions: the real gain and the masked estimate."""
    m = float(s @ x / (x @ x))
    return m, m * x


def figure1(path: Path) -> None:
    m, p = project(S, X)
    theta = np.arccos(S @ X / (np.linalg.norm(S) * np.linalg.norm(X)))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)

    line = np.stack([-0.45 * X, 1.35 * X])
    ax.plot(line[:, 0], line[:, 1], ls="--", color="0.55", lw=1.0, zorder=1)
    ax.annotate(r"$\mathbb{R}x$", 1.28 * X + (0.0, 0.09), color="0.4", fontsize=11)

    ax.fill([0, X[0], S[0]], [0, X[1], S[1]], color="0.9", zorder=1)
    for vec, name, off in ((X, "$x$", (0.04, -0.10)), (S, "$s$", (-0.13, 0.04))):
        ax.annotate("", vec, (0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.0, color="k"))
        ax.annotate(name, vec + off, fontsize=13)

    ax.plot(*p, "o", ms=7, mfc="w", mec="0.25", mew=1.6, zorder=4)
    ax.annotate(rf"$m^\star x$,  $m^\star={m:.2f}$", p + (-0.34, -0.21), fontsize=11, color="0.25")

    ax.plot([p[0], S[0]], [p[1], S[1]], color="crimson", lw=2.0)
    ax.annotate(
        rf"$|s|\sin\theta$ = {np.linalg.norm(S - p):.2f}",
        (p + S) / 2 + (0.05, 0.0),
        fontsize=11,
        color="crimson",
    )

    # right-angle tick at the foot of the projection
    u = X / np.linalg.norm(X)
    n = (S - p) / np.linalg.norm(S - p)
    corner = p + 0.075 * (u + n)
    ax.plot(*np.stack([p + 0.075 * u, corner, p + 0.075 * n]).T, color="crimson", lw=1.0)

    arc = np.linspace(0, theta, 40)
    rot = np.arctan2(X[1], X[0])
    ax.plot(0.36 * np.cos(arc + rot), 0.36 * np.sin(arc + rot), color="k", lw=1.0)
    ax.annotate(rf"$\theta={np.degrees(theta):.0f}^\circ$", (0.30, 0.26), fontsize=11)

    ceiling = -10 * np.log10(np.sin(theta) ** 2)
    ax.set_title(
        "A real mask reaches only the line $\\mathbb{R}x$\n"
        rf"per-bin ceiling of the class: {ceiling:.1f} dB",
        fontsize=11,
    )
    ax.set_xlim(-0.35, 2.05)
    ax.set_ylim(-0.30, 1.45)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


LADDER = (
    ("real mask\n$m\\,I_2$, 1 dof", np.array([[0.7, 0.0], [0.0, 0.7]]), np.zeros(2)),
    ("complex mask\nsimilitude, 2 dof", np.array([[0.62, -0.33], [0.33, 0.62]]), np.zeros(2)),
    ("widely linear\n$GL(2)$, 4 dof", np.array([[0.95, -0.30], [0.18, 0.42]]), np.zeros(2)),
    ("affine widely linear\n4 dof + offset", np.array([[0.95, -0.30], [0.18, 0.42]]), np.array([0.42, -0.30])),
)


def figure3(path: Path) -> None:
    t = np.linspace(0, 2 * np.pi, 256)
    circle = np.stack([np.cos(t), np.sin(t)])
    marks = np.stack([np.cos(t[::32]), np.sin(t[::32])])

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for ax, (name, a, c) in zip(axes, LADDER):
        img = a @ circle + c[:, None]
        ax.plot(*circle, color="0.75", lw=1.0, ls="--")
        ax.plot(*img, color="crimson", lw=2.0)
        for k in range(marks.shape[1]):
            src = marks[:, k]
            dst = a @ src + c
            ax.annotate("", dst, src, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.45"))
        ax.plot(0, 0, "k+", ms=8)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("0.85")
    fig.suptitle(
        "What each block structure does to the unit circle of one bin "
        "(grey: input, red: image, arrows: where directions go)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def demo() -> None:
    """The projection is orthogonal and the residual is the one of Proposition 1."""
    m, p = project(S, X)
    assert abs((S - p) @ X) < 1e-12, "residual not orthogonal to the mixture"
    theta = np.arccos(S @ X / (np.linalg.norm(S) * np.linalg.norm(X)))
    assert abs(np.linalg.norm(S - p) - np.linalg.norm(S) * np.sin(theta)) < 1e-12
    # a real mask cannot beat m*, checked against a dense sweep
    grid = np.linspace(-3, 3, 20001)
    err = ((S[None, :] - grid[:, None] * X[None, :]) ** 2).sum(axis=1)
    assert err.min() >= np.sum((S - p) ** 2) - 1e-9
    # the similitude preserves angles, the general linear map does not
    for name, a, _ in LADDER:
        u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        cos = (a @ u) @ (a @ v) / (np.linalg.norm(a @ u) * np.linalg.norm(a @ v))
        conformal = abs(cos) < 1e-12
        assert conformal == ("mask" in name), name
    print("demo ok")


if __name__ == "__main__":
    demo()
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    out.mkdir(parents=True, exist_ok=True)
    figure1(out / "fig1_mask_geometry.png")
    figure3(out / "fig3_block_ladder.png")
    print(f"wrote {out / 'fig1_mask_geometry.png'} and {out / 'fig3_block_ladder.png'}")
