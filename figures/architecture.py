"""Figure 1 of ankyra-lite: the issue times unrolled on one unit's record (a), with zoom-ins on the carrier network (b)
and the anchoring layer (c); for release 0.2.0 also the carrier's training (recipe Rw, six seeds; a strip of b) and the
origin-continuity layer after the anchoring layer (the last row of c).

    draw(path_stem)                      writes <path_stem>.pdf (vector, TrueType fonts embedded) and <path_stem>.png
                                         (300 dpi) and returns the layout problems found (an empty list when clean)
    build(n_parameters, cost_text, release="0.2")
                                         the figure itself (a matplotlib Figure, 7.2 in wide), for figures/make_figures.py;
                                         release "0.1" draws the 0.1.x figure (recipe R, three seeds, no continuation layer)
    layout_problems(fig)                 overlapping labels, labels outside their box or panel, text below 6 pt

Facts follow ankyra_lite/carrier/model.py (CopyPointerNetwork; build_carrier: 61,613 parameters, width 48, rank 8,
copy pointer over 56 days with 16-dimensional keys), features.py (token layout and the scale s), inference.py (issue
times o - 744k, k = 0..6, 1,344-hour contexts, the training cut), train.py (0.2.0: recipe Rw, the span return with the
inverse-noise loss weight w = max(sigma^2, 25% quantile)^(-1/2) per training window, six seeds averaged; 0.1.x: recipe
R, three seeds), ankyra_lite/anchor/core.py and analog.py (level, handover, within-day trust), anchor/readouts.py (peak
readout), ankyra_lite/continuation.py (0.2.0: F' = F exp(delta), delta(h) = kappa1 x_f exp(-h/12) + kappa2 x_d
exp(-h/168); state windows bypassed; energy, level and peak read before it, readout rule A) and ankyra_lite/forecast.py
(state windows: weekly seasonal naive).  The load record of panel (a) and every curve, bar and
weight in the glyphs are synthetic and illustrative (no observed data, no results).  Inside the glyphs the drawing is
consistent with the formulas: the carrier trajectory in (b) is the product 744 l p_d q_dh of the factors drawn beside
it, and the copy-pointer weights are a softmax with a day-type bonus over the 56 days drawn.  In (a) each pseudo-origin
forecast F(k) is drawn by its daily means, and the grey band is its gap to the observed daily means (the score e_k).
Nothing here compares ankyra-lite with ANKYRA; the anchoring layer is named as the copy of ANKYRA 2.2.1 it is.

The module has no state: it sets its rcParams only inside ``plt.rc_context(RC)`` and builds a stand-alone Figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

__all__ = ["draw", "build", "layout_problems", "RC", "N_PARAMETERS", "WIDTH", "HEIGHT"]

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial", "mathtext.fallback": "stixsans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
    "hatch.linewidth": 0.4,
}
N_PARAMETERS = 61_613

# palette of the repository (figures/make_figures.py) and light tints of the same hues
IN, IN_L, IN_M = "#4A6FA5", "#A9BEDD", "#7E9CC8"            # inputs
CARRIER, CAR_D, CAR_M, CAR_L, CAR_XL = "#1B7F79", "#0F4F4B", "#5FA9A3", "#A9D3CE", "#EAF5F3"   # carrier network
HIST, HIST_L, HIST_XL = "#C0762A", "#EBCDA9", "#FBF4EC"     # the unit's own history
LITE, LITE_L, LITE_XL = "#B2182B", "#E3A3AA", "#FBEFF0"     # anchoring layer and ankyra-lite
INK, GREY, GREY_M, FRAME = "#1F1F1F", "#595959", "#8C8C8C", "#C4C4C4"
OBS, OBS_E, OBS_D = "#D3DAE5", "#9AA8BF", "#4F5F7A"         # observed load: silhouette, its edge, its daily means
ERR = "#B9B9B9"                                              # error of a pseudo-origin forecast

WIDTH = 7.2                    # inches (double column); the drawing axes span the figure, one data unit = one inch
FS = 6.5                       # body text (pt)
FS_S = 6.0                     # smallest text
FS_M = 6.8                     # formulas
FS_T = 7.3                     # panel titles
FS_L = 9.0                     # panel letters
LW = 0.7                       # every connector
MS = 4.6                       # every arrowhead


# ============================================================================ drawing helpers
class Canvas:
    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.boxes = {}                 # name -> (x0, y0, x1, y1): the texts drawn inside must stay inside
        self._renderer = None

    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = self.fig.canvas.get_renderer()
        return self._renderer

    def width(self, artist):
        """Width of a text artist in inches."""
        return artist.get_window_extent(self.renderer).width / self.fig.dpi

    def rbox(self, x, y, w, h, fc, ec, lw=0.6, r=0.05, z=1, ls="-", name=None):
        if name:
            self.boxes[name] = (x, y, x + w, y + h)
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
        self.ax.add_patch(p)
        return p

    def text(self, x, y, s, fs=FS, color=INK, ha="left", va="center", z=7, **kw):
        return self.ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z, **kw)

    def segments(self, x, y, parts, fs=FS, va="center", z=7, ha="left"):
        """Consecutive text pieces on one line, each with its own colour and weight: [(text, colour, kwargs), ...]."""
        arts = [self.text(0, y, s, fs=fs, color=c, va=va, z=z, **kw) for s, c, kw in parts]
        widths = [self.width(a) for a in arts]
        if ha == "right":
            x -= sum(widths)
        elif ha == "center":
            x -= sum(widths) / 2
        for a, w in zip(arts, widths):
            a.set_x(x)
            x += w
        return x

    def arrow(self, p0, p1, color=GREY_M, lw=LW, ms=MS, z=3, ls="-"):
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0,
                            zorder=z, linestyle=ls, capstyle="butt", joinstyle="miter")
        self.ax.add_patch(a)
        return a

    def poly_arrow(self, pts, color=GREY_M, lw=LW, ms=MS, z=3):
        xs, ys = zip(*pts)
        self.ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=z, solid_capstyle="butt", solid_joinstyle="miter")
        self.arrow(pts[-2], pts[-1], color=color, lw=lw, ms=ms, z=z)

    def line(self, xs, ys, color=GREY_M, lw=0.6, z=2, **kw):
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=z, **kw)

    def token(self, x, y, w, h, color, cells=4, z=4, ec="white", lw=0.35):
        """A token drawn as a short column vector (stacked cells)."""
        ch = h / cells
        for i in range(cells):
            self.ax.add_patch(Rectangle((x, y + i * ch), w, ch, fc=color, ec=ec, lw=lw, zorder=z))

    def curve(self, x, y, w, h, v, color, lw=0.7, z=5, lo=None, hi=None, **kw):
        v = np.asarray(v, float)
        lo = v.min() if lo is None else lo
        hi = v.max() if hi is None else hi
        xs = x + w * np.linspace(0, 1, len(v))
        self.ax.plot(xs, y + h * (v - lo) / max(hi - lo, 1e-9), color=color, lw=lw, zorder=z, solid_capstyle="round",
                     solid_joinstyle="round", **kw)

    def letter(self, x, y, letter, title, sub=None):
        self.text(x, y, letter, fs=FS_L, fontweight="bold")
        t = self.text(x + 0.17, y, title, fs=FS_T, fontweight="bold")
        if sub:
            self.text(x + 0.17 + self.width(t) + 0.10, y - 0.003, sub, fs=FS_S, color=GREY)

    def tag(self, x, y, s, color=LITE, fs=FS_S, ha="right"):
        """A small rounded label (the evidence of an anchoring block)."""
        return self.text(x, y, s, fs=fs, color=color, ha=ha,
                         bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.45", fc="white", ec=color, lw=0.5))


def layout_problems(fig, tol=0.004):
    """Overlapping text labels, labels outside the figure or outside the innermost registered box that holds their
    centre, and text set below 6 pt.  Returns a list of strings (empty when the layout is clean)."""
    canvas = getattr(fig, "_arch_canvas", None)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.axes[0].texts:
        s = t.get_text()
        if not s.strip():
            continue
        bb = t.get_window_extent(r)
        items.append((s, bb.x0 / fig.dpi, bb.y0 / fig.dpi, bb.x1 / fig.dpi, bb.y1 / fig.dpi, t.get_fontsize()))
    out = []
    W_, H_ = fig.get_size_inches()
    for i, (s, x0, y0, x1, y1, fs) in enumerate(items):
        if fs < 6.0 - 1e-9:
            out.append(f"below 6 pt ({fs:.1f}): {s!r}")
        if x0 < -tol or y0 < -tol or x1 > W_ + tol or y1 > H_ + tol:
            out.append(f"outside the figure: {s!r}")
        for s2, a0, b0, a1, b1, _ in items[i + 1:]:
            if min(x1, a1) - max(x0, a0) > tol and min(y1, b1) - max(y0, b0) > tol:
                out.append(f"overlap: {s!r} / {s2!r}")
        if canvas is not None:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            holders = sorted(((bx1 - bx0) * (by1 - by0), name, (bx0, by0, bx1, by1))
                             for name, (bx0, by0, bx1, by1) in canvas.boxes.items()
                             if bx0 <= cx <= bx1 and by0 <= cy <= by1)
            if holders:
                _, name, (bx0, by0, bx1, by1) = holders[0]
                if x0 < bx0 - tol or x1 > bx1 + tol or y0 < by0 - tol or y1 > by1 + tol:
                    out.append(f"outside its box {name}: {s!r} ({x0:.3f}-{x1:.3f}, {y0:.3f}-{y1:.3f})")
    return out


# ============================================================================ synthetic, illustrative load
DELTA = 744                     # hours between issue times (31 days)


def _office_day(h, workday, amp):
    """Smooth office-like day: base load plus an occupied-hours plateau."""
    rise = 1 / (1 + np.exp(-(h - 7.2) * 1.6))
    fall = 1 / (1 + np.exp((h - 18.3) * 1.3))
    bump = rise * fall * (1 + 0.10 * np.sin(np.pi * (h - 7) / 11.5))
    return 0.42 + np.where(workday, amp, 0.07 * amp) * bump


def synthetic_series(seed=7):
    """Hourly load from -7 * 744 h to +744 h around the origin (index 0 = the origin); day 0 is a Thursday."""
    rng = np.random.default_rng(seed)
    t = np.arange(-7 * DELTA, DELTA)
    day = np.floor_divide(t, 24)
    h = t - 24 * day
    weekday = (day + 3) % 7                                                      # Monday = 0
    workday = (weekday < 5) & ~np.isin(day, [-5 * 31 + 9, -2 * 31 + 18])        # two weekday holidays
    season = 0.5 - 0.5 * np.cos(2 * np.pi * (t + 7 * DELTA) / (16 * DELTA))       # slow cooling season
    amp = 0.45 + 0.42 * season
    shift = np.where(t >= -3 * DELTA + 200, 0.06, 0.0)                            # an occupancy change
    dscale = 1 + 0.045 * rng.standard_normal(day.max() - day.min() + 1)[day - day.min()]
    noise = np.zeros(len(t))
    e = rng.standard_normal(len(t)) * 0.028
    for i in range(1, len(t)):
        noise[i] = 0.72 * noise[i - 1] + e[i]
    y = (_office_day(h, workday, amp) + shift) * dscale + noise
    return t, np.clip(y, 0.05, None), workday, weekday


def profile_forecast(t, y, workday, q, level_err):
    """A smooth 31-day stand-in forecast issued at q: the workday / non-workday profile of the 28 days before q."""
    pre = (t >= q - 28 * 24) & (t < q)
    hh = t % 24
    prof = {}
    for wd in (True, False):
        m = pre & (workday == wd)
        prof[wd] = np.array([y[m & (hh == k)].mean() for k in range(24)])
    sel = (t >= q) & (t < q + DELTA)
    f = np.array([prof[bool(w)][k] for w, k in zip(workday[sel], hh[sel])])
    return t[sel], f * (1 + level_err)


def decoder_parts(seed=3):
    """Illustrative factors of the energy-budget decoder for the delivered month: l, p_d, q_dh and F = 744 l p q."""
    rng = np.random.default_rng(seed)
    d = np.arange(31)
    work = ((d + 3) % 7) < 5
    p = np.where(work, 1.0, 0.56) * (1 + 0.035 * rng.standard_normal(31)) * (1 + 0.05 * np.sin(d / 5))
    p = p / p.sum()
    hh = np.arange(24) + 0.5
    qw = _office_day(hh, True, 0.95)
    qe = _office_day(hh, False, 0.95)
    q = np.where(work[:, None], qw, qe) * (1 + 0.03 * rng.standard_normal((31, 1)) * np.cos(2 * np.pi * (hh - 13) / 24))
    q = q / q.sum(1, keepdims=True)
    level = 0.80
    F = 744 * level * p[:, None] * q
    return level, p, q, F, work


def pointer_weights(weekday, workday, target_weekday, seed=11):
    """Copy-pointer weights over the last 56 days: a softmax of a similarity score plus a bonus for the same day type."""
    rng = np.random.default_rng(seed)
    match = (weekday == target_weekday) & workday                  # day types: weekday 0..6, holiday 7
    score = 1.25 * workday + 2.4 * match + 0.012 * np.arange(len(weekday)) + 0.22 * rng.standard_normal(len(weekday))
    a = np.exp(score - score.max())
    return a / a.sum(), match


# ============================================================================ geometry (inches)
XL, XR = 0.16, 7.15                                # time axis of panel (a)
C_LEFT, R_RIGHT = 0.98, 0.80                       # left context and right margin, in units of DELTA
T_MIN, T_MAX = -(6 + C_LEFT) * DELTA, (1 + R_RIGHT) * DELTA


def X(t):
    return XL + (XR - XL) * (np.asarray(t, float) - T_MIN) / (T_MAX - T_MIN)


DLT = float(X(DELTA) - X(0))                      # one month in inches
PITCH, BAR_H = 0.128, 0.098                        # lanes of panel (a)
CHIP_PAD = 0.035
LANE0_GAP = 0.06                                   # extra room between F(0) and the anchoring box below it
TOP_BC_010 = 3.30                                  # top of panels (b) and (c), release 0.1.x
EXTRA_020 = 0.66                                   # release 0.2.0: room for the training strip (b) and the continuation row (c)
RELEASES = ("0.1", "0.2")
RELEASE = "0.2"                                    # the release drawn (set by build; 0.2 = ankyra-lite 0.2.0)


def _geometry(release):
    """Set the module geometry for a release: 0.1 (recipe R, three seeds, no continuation layer) or 0.2 (recipe Rw,
    six seeds, the origin-continuity layer after the anchoring layer); panels (b) and (c) are taller in 0.2."""
    global RELEASE, TOP_BC, PB, PC, Y_LANE6, Y_LANE0, Y_D0, Y_D1, HEIGHT
    if release not in RELEASES:
        raise ValueError(f"release must be one of {RELEASES}, not {release!r}")
    RELEASE = release
    TOP_BC = TOP_BC_010 + (EXTRA_020 if release == "0.2" else 0.0)
    PB = (0.06, 0.08, 4.12, TOP_BC)                # panel (b): x0, y0, x1, y1
    PC = (4.22, 0.08, 7.14, TOP_BC)                # panel (c)
    Y_LANE6 = TOP_BC + 0.25 + BAR_H / 2 + CHIP_PAD
    Y_LANE0 = Y_LANE6 + 6 * PITCH + LANE0_GAP
    Y_D0 = Y_LANE0 + 0.40                          # data layer
    Y_D1 = Y_D0 + 0.66
    HEIGHT = Y_D1 + 0.46


_geometry(RELEASE)


def lane_y(k):
    return Y_LANE0 - k * PITCH - (LANE0_GAP if k else 0.0)


# ============================================================================ the figure
def build(n_parameters=N_PARAMETERS, cost_text=None, release="0.2"):
    """The figure (call inside ``plt.rc_context(RC)``; :func:`draw` does).  ``cost_text`` is an optional line on the
    time per forecast (figures/make_figures.py reads it from results/cost_per_window.csv).  ``release``: "0.2"
    (ankyra-lite 0.2.0: recipe Rw, six seeds, the origin-continuity layer) or "0.1" (0.1.x: recipe R, three seeds)."""
    _geometry(release)
    fig = Figure(figsize=(WIDTH, HEIGHT), dpi=300)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    c = Canvas(fig, ax)
    fig._arch_canvas = c
    t, y, workday, weekday = synthetic_series()
    _panel_a(c, t, y, workday)
    _panel_b(c, t, y, workday, weekday, n_parameters)
    _panel_c(c, cost_text)
    return fig


def draw(path_stem, n_parameters=N_PARAMETERS, cost_text=None, check=True, release="0.2"):
    """Write <path_stem>.pdf and <path_stem>.png (300 dpi); return the layout problems (empty when clean)."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig = build(n_parameters, cost_text, release=release)
        problems = layout_problems(fig) if check else []
        fig.savefig(str(path_stem) + ".pdf", metadata={"CreationDate": None})
        fig.savefig(str(path_stem) + ".png", dpi=300)
    return problems


# ---------------------------------------------------------------------------- panel (a): issue times unrolled
def _panel_a(c, t, y, workday):
    ax = c.ax
    c.boxes["panel a"] = (0.0, TOP_BC + 0.05, WIDTH, HEIGHT)
    yl = HEIGHT - 0.11
    c.letter(0.03, yl, "a", "Issue times on the unit's record")

    # inline legend on the title row (right-aligned)
    parts = [("observed load (synthetic)", "obs"), ("carrier $F^{(k)}$", "car"), ("error $e_k$", "err"),
             ("ankyra-lite", "lite")]
    labels = [c.text(0, yl, s, fs=FS, color=INK) for s, _ in parts]
    widths = [c.width(a) for a in labels]
    sw, gap_in, gap_out = 0.19, 0.06, 0.17
    x = XR - (sum(widths) + len(parts) * (sw + gap_in) + (len(parts) - 1) * gap_out)
    for (s, kind), lab, w_ in zip(parts, labels, widths):
        if kind == "obs":
            ax.add_patch(Rectangle((x, yl - 0.04), sw, 0.08, fc=OBS, ec=OBS_E, lw=0.4, zorder=5))
            c.line([x, x + sw], [yl, yl], color=OBS_D, lw=0.7, z=6)
        elif kind == "err":
            ax.add_patch(Rectangle((x, yl - 0.04), sw, 0.08, fc=ERR, ec="none", zorder=5))
        else:
            c.line([x, x + sw], [yl, yl], color=CARRIER if kind == "car" else LITE, lw=1.1, z=5)
        lab.set_x(x + sw + gap_in)
        x += sw + gap_in + w_ + gap_out

    ymax = 1.62
    sy = lambda v: Y_D0 + (Y_D1 - Y_D0) * np.asarray(v) / ymax                       # noqa: E731

    # delivered month background and issue-time guides
    ax.add_patch(Rectangle((X(0), Y_D0), DLT, Y_D1 - Y_D0 + 0.03, fc=LITE_XL, ec="none", zorder=0))
    for k in range(-1, 7):
        xk = X(-k * DELTA)
        y_lo = lane_y(max(k, 0)) - BAR_H / 2
        c.line([xk, xk], [y_lo, Y_D1 + 0.03], color="#CBCBCB", lw=0.45, z=1, ls=(0, (1.2, 1.6)))

    # observed load before the origin (silhouette, the base layer); earlier history faded
    obs = t < 0
    xo, yo = X(t[obs]), sy(y[obs])
    early = t[obs] < -6 * DELTA
    ax.fill_between(xo, Y_D0, yo, where=~early | (t[obs] == -6 * DELTA), color=OBS, lw=0, zorder=2)
    ax.fill_between(xo, Y_D0, yo, where=early | (t[obs] == -6 * DELTA), color="#EAEDF2", lw=0, zorder=2)
    ax.plot(xo[~early], yo[~early], color=OBS_E, lw=0.2, zorder=2)
    ax.plot(xo[early], yo[early], color="#D3D8E1", lw=0.2, zorder=2)
    c.line([X(T_MIN), X(T_MAX)], [Y_D0, Y_D0], color=GREY_M, lw=0.5, z=3)

    # each pseudo-origin forecast F(k) against the observed month it covers (daily means; grey = its error e_k)
    errs = {1: 0.07, 2: -0.08, 3: -0.10, 4: 0.06, 5: -0.06, 6: 0.08}
    for k, e in errs.items():
        tf, f = profile_forecast(t, y, workday, -k * DELTA, e)
        fd = f.reshape(31, 24).mean(1)
        od = y[(t >= -k * DELTA) & (t < -(k - 1) * DELTA)].reshape(31, 24).mean(1)
        xd = X(-k * DELTA + 12 + 24 * np.arange(31))
        ax.fill_between(xd, sy(od), sy(fd), color=ERR, lw=0, zorder=3, alpha=0.95)
        ax.plot(xd, sy(od), color=OBS_D, lw=0.55, zorder=4)
        ax.plot(xd, sy(fd), color=CARRIER, lw=0.8, zorder=5)
    tf, f0 = profile_forecast(t, y, workday, 0, 0.0)
    fl = f0 * (1 + 0.02 * np.sin(np.linspace(0, 3, len(f0))))
    ax.plot(X(tf), sy(fl), color=LITE, lw=0.35, zorder=5)

    # brackets above the data layer
    yb = Y_D1 + 0.10

    def bracket(x0, x1, s, color):
        c.line([x0, x0, x1, x1], [yb - 0.035, yb, yb, yb - 0.035], color=color, lw=0.55, z=6)
        c.text((x0 + x1) / 2, yb + 0.085, s, fs=FS, color=color, ha="center")

    bracket(X(-6 * DELTA) + 0.03, X(0) - 0.04,
            "up to six months already observed at $o$: each carrier forecast is scored on its own month", GREY)
    bracket(X(0) + 0.04, X(DELTA) - 0.03, "delivered month", LITE)
    c.text(XR, yb + 0.085, r"$\Delta$ = 744 h", fs=FS_S, color=GREY, ha="right")
    c.text(X(-6.49 * DELTA), Y_D1 - 0.04, "earlier\nhistory", fs=FS_S, color=GREY_M, ha="center", va="top",
           linespacing=1.1)

    # origin, issue-time labels and the score of each month
    c.line([X(0), X(0)], [lane_y(0) - BAR_H / 2, Y_D1 + 0.03], color=LITE, lw=0.9, z=6)
    ya = Y_D0 - 0.105
    for k in range(6, 0, -1):
        c.text(X(-k * DELTA), ya, f"$o-{k}\\Delta$" if k > 1 else "$o-\\Delta$", fs=FS_S, color=GREY, ha="center",
               bbox=dict(fc="white", ec="none", pad=0.6))
        c.text(X(-(k - 0.5) * DELTA), ya, f"$e_{k}$", fs=FS, color=GREY, ha="center")
    c.text(X(0), ya, "origin $o$", fs=FS, color=LITE, ha="center", fontweight="bold",
           bbox=dict(fc="white", ec="none", pad=0.8))
    c.text(X(DELTA), ya, "$o+\\Delta$", fs=FS_S, color=GREY, ha="center", bbox=dict(fc="white", ec="none", pad=0.6))

    # carrier chip (zoomed in panel b)
    chip_x0, chip_x1 = X(T_MIN) + 0.01, X(-6 * DELTA) - 0.075
    chip_y0, chip_y1 = lane_y(6) - BAR_H / 2 - CHIP_PAD, lane_y(0) + BAR_H / 2 + CHIP_PAD
    c.rbox(chip_x0, chip_y0, chip_x1 - chip_x0, chip_y1 - chip_y0, CAR_XL, CARRIER, lw=0.8, r=0.05, z=5, name="chip")
    cx = (chip_x0 + chip_x1) / 2
    cym = (chip_y0 + chip_y1) / 2 + 0.06
    c.text(cx, cym + 0.07, "carrier\nnetwork", fs=FS + 0.3, color=CARRIER, ha="center", fontweight="bold",
           linespacing=1.05)
    c.text(cx, cym - 0.13, "6 seeds · Rw" if RELEASE == "0.2" else "3 seeds", fs=FS_S, color=CARRIER, ha="center")
    c.text(cx, chip_y0 + 0.10, "b", fs=FS + 1.0, color=CARRIER, ha="center", fontweight="bold")

    # anchoring box (zoomed in panel c); in 0.2.0 followed by the continuation layer (also zoomed in panel c)
    bx0, bx1 = X(0) + 0.10, X(T_MAX) - 0.005
    by0, by1 = lane_y(6) - BAR_H / 2 - CHIP_PAD, lane_y(1) + BAR_H / 2 + 0.012
    bym = (by0 + by1) / 2 + 0.06
    if RELEASE != "0.2":
        c.rbox(bx0, by0, bx1 - bx0, by1 - by0, LITE_XL, LITE, lw=0.8, r=0.05, z=5, name="anchor box")
        bxc = (bx0 + bx1) / 2
        c.text(bxc, bym + 0.07, "anchoring layer", fs=FS + 0.3, color=LITE, ha="center", fontweight="bold")
        c.text(bxc, bym - 0.07, "copy of ANKYRA 2.2.1", fs=FS_S, color=LITE, ha="center")
        c.text(bxc, by0 + 0.10, "c", fs=FS + 1.0, color=LITE, ha="center", fontweight="bold")
        xo_ = X(DELTA) + 0.33                                      # the delivered month leaves the anchoring box here
    else:
        ax1 = X(0.86 * DELTA)                                      # right end of the anchoring box
        c.rbox(bx0, by0, ax1 - bx0, by1 - by0, LITE_XL, LITE, lw=0.8, r=0.05, z=5, name="anchor box")
        bxc = (bx0 + ax1) / 2
        c.text(bxc, bym + 0.09, "anchoring\nlayer", fs=FS + 0.3, color=LITE, ha="center", fontweight="bold", linespacing=1.0)
        c.text(bxc, bym - 0.14, "copy of\nANKYRA 2.2.1", fs=FS_S, color=LITE, ha="center", linespacing=1.05)
        c.text(bxc, by0 + 0.10, "c", fs=FS + 1.0, color=LITE, ha="center", fontweight="bold")
        kx0 = ax1 + 0.12
        c.rbox(kx0, by0, bx1 - kx0, by1 - by0, "white", LITE, lw=0.8, r=0.05, z=5, name="continuation box")
        kxc = (kx0 + bx1) / 2
        c.text(kxc, bym + 0.09, "continuation\nlayer", fs=FS + 0.3, color=LITE, ha="center", fontweight="bold", linespacing=1.0)
        c.text(kxc, bym - 0.14, "4 frozen\nconstants", fs=FS_S, color=LITE, ha="center", linespacing=1.05)
        c.text(kxc, by0 + 0.10, "c", fs=FS + 1.0, color=LITE, ha="center", fontweight="bold")
        c.arrow((ax1 + 0.004, by0 + 0.10), (kx0 - 0.004, by0 + 0.10), color=LITE, lw=0.85)
        xo_ = kxc

    # lanes: issue (teal), forecast month (bar), score (grey) flowing into the anchoring layer
    for k in range(0, 7):
        yk = lane_y(k)
        xs, xe = X(-k * DELTA), X(-(k - 1) * DELTA)
        c.line([chip_x1, xs - 0.02], [yk, yk], color=CARRIER, lw=0.55, z=3, alpha=0.55)
        ax.plot([xs - 0.012], [yk], marker=">", ms=2.6, color=CARRIER, zorder=6, mec="none")
        c.rbox(xs + 0.012, yk - BAR_H / 2, xe - xs - 0.024, BAR_H, CARRIER if k else "white", CARRIER, lw=0.6,
               r=0.025, z=5)
        c.text((xs + xe) / 2, yk - 0.002, f"$F^{{({k})}}$", fs=FS, color="white" if k else CARRIER,
               ha="center", va="center", z=8)
        if k:
            c.arrow((xe - 0.005, yk), (bx0 - 0.004, yk), color="#7A7A7A", lw=0.6, z=3)
            ax.plot([xe - 0.012], [yk], marker="o", ms=2.4, color="white", mec=GREY, mew=0.55, zorder=6)
    c.text(X(-4 * DELTA) + 0.12, (lane_y(5) + lane_y(6)) / 2, "score $e_k$ of $F^{(k)}$ on its observed month",
           fs=FS_S, color=GREY, style="italic")
    c.text(chip_x1 + 0.08, lane_y(0) + 0.08, "issued at $o-k\\Delta$ from the data before the issue time",
           fs=FS_S, color=CARRIER, style="italic")
    # F(0) into the anchoring layer, and the delivered month out of it
    c.arrow((X(0.45 * DELTA), lane_y(0) - BAR_H / 2 - 0.004), (X(0.45 * DELTA), by1 + 0.002), color=CARRIER, lw=LW)
    c.poly_arrow([(xo_, by1 + 0.002), (xo_, sy(1.10)), (X(DELTA) + 0.03, sy(1.10))], color=LITE, lw=0.85)

    # zoom lines
    for (xa, xb) in ((chip_x0, PB[0]), (chip_x1, PB[2])):
        c.line([xa, xb], [chip_y0, PB[3]], color=GREY_M, lw=0.45, z=1, ls=(0, (2.5, 2)))
    for (xa, xb) in ((bx0, PC[0]), (bx1, PC[2])):
        c.line([xa, xb], [by0, PC[3]], color=GREY_M, lw=0.45, z=1, ls=(0, (2.5, 2)))


# ---------------------------------------------------------------------------- panel (b): the carrier network
def _panel_b(c, t, y, workday, weekday, n_parameters):
    ax = c.ax
    x0, y0, x1, y1 = PB
    c.rbox(x0, y0, x1 - x0, y1 - y0, "white", FRAME, lw=0.5, r=0.06, z=0, name="panel b")
    c.letter(x0 + 0.08, y1 - 0.13, "b", "Carrier network",
             f"CopyPointerNetwork, {n_parameters:,} parameters, width 48")
    lx0, lx1 = 0.14, 2.30                         # left column (history side)
    rx0, rx1 = 2.74, 4.02                         # right column (calendar side)

    # ---- input tokens: 24 monthly (older), 366 daily, oldest to newest; 31 future days
    tw, th, tp = 0.058, 0.16, 0.082               # token width, height, pitch
    yt = 0.34
    xs0 = 0.22
    xm = [xs0 + i * tp for i in (0, 1, 3, 4)]
    for xx in xm:
        c.token(xx, yt, tw, th, IN_L)
    c.text(xs0 + 2 * tp + tw / 2, yt + th / 2, "…", fs=FS, color=GREY, ha="center")
    xd0 = xs0 + 5 * tp + 0.06
    xd = [xd0 + i * tp for i in (0, 1)] + [xd0 + (i + 3) * tp for i in range(15)]
    for xx in xd:
        c.token(xx, yt, tw, th, IN)
    c.text(xd0 + 2 * tp + tw / 2, yt + th / 2, "…", fs=FS, color=GREY, ha="center")
    l56 = xd[-4:]
    c.rbox(l56[0] - 0.018, yt - 0.022, l56[-1] + tw - l56[0] + 0.036, th + 0.044, "none", HIST, lw=0.8, r=0.025, z=6)
    ylab = 0.20
    c.text((xm[0] + xm[-1] + tw) / 2, ylab, r"$X^{\mathrm{mon}}\in\mathbb{R}^{24\times16}$", fs=FS, ha="center")
    c.text((xd[0] + l56[0]) / 2 + 0.02, ylab, r"$X^{\mathrm{day}}\in\mathbb{R}^{366\times69}$", fs=FS, ha="center")
    c.text((l56[0] + l56[-1] + tw) / 2, ylab, "last 56 d", fs=FS_S, color=HIST, ha="center")
    xf = [rx0 + 0.02 + i * tp for i in range(6)] + [rx0 + 0.02 + (i + 7) * tp for i in range(2)]
    for xx in xf:
        c.token(xx, yt, tw, th, IN_M)
    c.text(rx0 + 0.02 + 6 * tp + tw / 2, yt + th / 2, "…", fs=FS, color=GREY, ha="center")
    c.text((xf[0] + xf[-1] + tw) / 2, ylab, r"$X^{\mathrm{fut}}\in\mathbb{R}^{31\times16}$", fs=FS, ha="center")

    # ---- encoder: shared gated causal blocks, a dilated stack over the newest day tokens
    ey0, ey1 = 0.58, 1.15
    c.rbox(lx0, ey0, lx1 - lx0, ey1 - ey0, CAR_XL, CAR_L, lw=0.6, r=0.04, z=1, name="encoder")
    c.segments(lx0 + 0.07, ey1 - 0.085, [("Encoder", CARRIER, {"fontweight": "bold"}),
                                         ("   2 shared gated causal blocks", GREY, {})], fs=FS)
    c.text(lx0 + 0.07, ey1 - 0.195, "kernel 3; dilations 1…128 (days), 1…8 (months)",
           fs=FS_S, color=GREY)
    cols = [xx + tw / 2 for xx in xd[2:]]           # 15 contiguous day tokens
    lev_y = [ey0 + 0.05 + i * 0.080 for i in range(4)]
    used = [{14}]
    for d in (4, 2, 1):
        used.append({u - j * d for u in used[-1] for j in range(3) if u - j * d >= 0})
    for li in range(4):
        ax.plot(cols, [lev_y[li]] * 15, "o", ms=1.4, color="#C5DFDC", zorder=3, mec="none")
    for li, d in zip(range(3, 0, -1), (4, 2, 1)):
        for u in used[3 - li]:
            for j in range(3):
                v = u - j * d
                if v >= 0:
                    c.line([cols[u], cols[v]], [lev_y[li], lev_y[li - 1]], color=CARRIER, lw=0.4, z=3, alpha=0.75)
    for li in range(4):
        pts = sorted(used[3 - li])
        ax.plot([cols[p] for p in pts], [lev_y[li]] * len(pts), "o", ms=1.9, color=CARRIER, zorder=4, mec="none")
    for li, d in zip(range(1, 4), (1, 2, 4)):
        c.text(cols[0] - 0.07, (lev_y[li] + lev_y[li - 1]) / 2, f"$d={d}$", fs=FS_S, color=CARRIER, ha="right")
    c.arrow((cols[2], yt + th + 0.012), (cols[2], ey0 - 0.003), color=IN)
    c.arrow((xs0 + 2 * tp + tw / 2, yt + th + 0.012), (xs0 + 2 * tp + tw / 2, ey0 - 0.003), color=IN)

    # ---- memory H (24 month + 366 day memories)
    ym, mh = 1.23, 0.10
    for xx in xm:
        c.token(xx, ym, tw, mh, CAR_L, cells=3)
    for xx in xd:
        c.token(xx, ym, tw, mh, CARRIER, cells=3)
    c.text(xs0 + 2 * tp + tw / 2, ym + mh / 2, "…", fs=FS, color=GREY, ha="center")
    c.text(xd0 + 2 * tp + tw / 2, ym + mh / 2, "…", fs=FS, color=GREY, ha="center")
    c.text(xs0, ym + mh + 0.085, r"memory $H\in\mathbb{R}^{390\times48}$", fs=FS, color=CARRIER)
    c.arrow((cols[6], ey1 + 0.004), (cols[6], ym - 0.004), color=CARRIER)

    # ---- right column: future tokens -> day queries (MLP) and the learned global query -> attention read-out
    yq, qh = 0.86, 0.10
    qx = [rx0 + 0.02 + i * tp for i in range(6)] + [rx0 + 0.02 + 7 * tp]
    for xx in qx:
        c.token(xx, yq, tw, qh, CAR_L, cells=3)
    c.text(rx0 + 0.02 + 6 * tp + tw / 2, yq + qh / 2, "…", fs=FS, color=GREY, ha="center")
    xmid = (xf[0] + xf[5] + tw) / 2
    c.arrow((xmid, yt + th + 0.012), (xmid, yq - 0.004), color=IN)
    c.text(xmid + 0.05, (yt + th + yq) / 2, "MLP", fs=FS_S, color=GREY)
    vx = rx1 - 0.20
    c.token(vx, yq, tw, qh, CAR_D, cells=3)
    c.text(vx + tw / 2, yq - 0.075, "$v$ (learned)", fs=FS_S, color=CAR_D, ha="center")

    ay0, ay1 = 1.17, 2.07
    c.rbox(rx0, ay0, rx1 - rx0, ay1 - ay0, CAR_XL, CAR_L, lw=0.6, r=0.04, z=1, name="read-out")
    c.text(rx0 + 0.07, ay1 - 0.09, "Attention read-out", fs=FS, color=CARRIER, fontweight="bold")
    qa = (qx[0] + qx[5] + tw) / 2
    c.arrow((qa, yq + qh + 0.004), (qa, ay0 - 0.003), color=CARRIER)
    c.text(qa + 0.05, (yq + qh + ay0) / 2, "31 day queries", fs=FS_S, color=CARRIER)
    c.arrow((vx + tw / 2, yq + qh + 0.004), (vx + tw / 2, ay0 - 0.003), color=CAR_D)
    # attention maps of the four heads (32 queries x 390 memories), schematic
    rng = np.random.default_rng(5)
    hx0, hy0, hw, hh_ = rx0 + 0.09, ay0 + 0.09, 0.50, 0.24
    for i in range(4):
        ox, oy = 0.045 * (3 - i), 0.035 * (3 - i)
        z = rng.random((8, 26)) ** 3
        z[:, -6:] += 0.6 * rng.random((8, 6))
        ax.imshow(z, extent=(hx0 + ox, hx0 + ox + hw - 0.135, hy0 + oy, hy0 + oy + hh_ - 0.105), cmap="BuGn",
                  vmin=0, vmax=1.4, aspect="auto", interpolation="nearest", zorder=3 + 0.1 * i)
        ax.add_patch(Rectangle((hx0 + ox, hy0 + oy), hw - 0.135, hh_ - 0.105, fc="none", ec=CARRIER, lw=0.4,
                               zorder=3.05 + 0.1 * i))
    c.text(hx0 + hw + 0.04, hy0 + 0.13, "4 heads", fs=FS_S, color=GREY)
    c.text(hx0 + hw + 0.04, hy0 + 0.035, r"$32\times390$", fs=FS_S, color=GREY)
    # read-out outputs: 31 day embeddings z_d and the window embedding v
    zy, zh = 1.68, 0.10
    zx = [rx0 + 0.09 + i * 0.05 for i in range(6)] + [rx0 + 0.09 + 7 * 0.05]
    for xx in zx:
        c.token(xx, zy, 0.036, zh, CAR_M, cells=3)
    c.text(rx0 + 0.09 + 6 * 0.05 + 0.018, zy + zh / 2, "…", fs=FS_S, color=GREY, ha="center")
    c.token(zx[-1] + 0.09, zy, 0.036, zh, CAR_D, cells=3)
    c.text(zx[-1] + 0.165, zy + zh / 2, r"$z_1\ldots z_{31},\ v$", fs=FS_S, color=CARRIER)
    c.arrow((xd[-1] + tw + 0.03, ym + mh / 2), (rx0 - 0.003, ym + mh / 2), color=CARRIER)
    c.text((xd[-1] + tw + rx0) / 2 + 0.02, ym + mh / 2 + 0.075, "K, V", fs=FS_S, color=CARRIER, ha="center")

    # ---- copy pointer over the unit's last 56 complete days
    py0, py1 = 1.50, 2.07
    c.rbox(lx0, py0, lx1 - lx0, py1 - py0, HIST_XL, HIST_L, lw=0.6, r=0.04, z=1, name="pointer")
    c.segments(lx0 + 0.07, py1 - 0.09, [("Copy pointer", CARRIER, {"fontweight": "bold"}),
                                        ("   last 56 complete days", HIST, {})], fs=FS)
    d_star = 5                                     # the forecast day followed through (b): a Tuesday
    target = (d_star + 3) % 7
    last = (t >= -56 * 24) & (t < 0)
    load56 = y[last]
    wd56 = weekday[last][::24]
    work56 = workday[last][::24]
    alpha, match = pointer_weights(wd56, work56, target)
    sx0, sx1 = lx0 + 0.25, lx1 - 0.07
    dw = (sx1 - sx0) / 56
    ly0, ly1 = py0 + 0.07, py0 + 0.21              # load strip
    ab0, ab1 = py0 + 0.25, py0 + 0.42             # attention weights
    for j in np.flatnonzero(match):
        ax.add_patch(Rectangle((sx0 + j * dw, ly0 - 0.01), dw, ab1 - ly0 + 0.02, fc=HIST_L, ec="none", alpha=0.55,
                               zorder=1.5))
    c.curve(sx0, ly0, sx1 - sx0, ly1 - ly0, load56, HIST, lw=0.35, lo=0.0)
    c.line([sx0, sx1], [ly0, ly0], color=GREY_M, lw=0.4, z=3)
    for j, a in enumerate(alpha):
        ax.add_patch(Rectangle((sx0 + j * dw + 0.15 * dw, ab0), 0.7 * dw, (ab1 - ab0) * a / alpha.max(), fc=CARRIER,
                               ec="none", zorder=4))
    c.line([sx0, sx1], [ab0, ab0], color=GREY_M, lw=0.4, z=3)
    c.text(sx0 - 0.04, (ab0 + ab1) / 2, r"$\alpha_{d,j}$", fs=FS, color=CARRIER, ha="right")
    c.text(sx0 - 0.04, (ly0 + ly1) / 2, r"$y_j$", fs=FS, color=HIST, ha="right")
    # pointer inputs: the query z_d from the read-out, 16-d keys from the day memories
    zq = zy + zh / 2 + 0.02
    c.arrow((rx0 - 0.004, zq), (lx1 + 0.004, zq), color=CARRIER)
    c.text((lx1 + rx0) / 2, zq + 0.07, "$z_d$", fs=FS, color=CARRIER, ha="center")
    c.text((lx1 + rx0) / 2, zq - 0.075, "query", fs=FS_S, color=CARRIER, ha="center")
    xk = (l56[0] + l56[-1] + tw) / 2
    c.arrow((xk, ym + mh + 0.004), (xk, py0 - 0.003), color=CARRIER)
    c.text(xk - 0.05, ym + mh + 0.085, "16-d keys", fs=FS_S, color=CARRIER, ha="right")

    # ---- energy-budget decoder: 744 l x p_d x q_dh = F_dh, one forecast day (dark) followed through every factor
    dy0, dy1 = 2.24, 3.06
    c.rbox(lx0, dy0, rx1 - lx0, dy1 - dy0, CAR_XL, CAR_L, lw=0.6, r=0.04, z=1, name="decoder")
    c.text(lx0 + 0.07, dy1 - 0.09, "Energy-budget decoder", fs=FS, fontweight="bold", color=CARRIER)
    c.text(rx1 - 0.07, dy1 - 0.09, r"$F_{d,h}=744\,\ell\,p_d\,q_{d,h}\geq0$", fs=FS_M, ha="right")
    level, p, q, F, work = decoder_parts()
    gy0, gy1 = dy0 + 0.25, dy0 + 0.60
    gh = gy1 - gy0
    yl1, yl2 = dy0 + 0.165, dy0 + 0.065            # label rows: symbol, description
    # scale s -> level
    sxc = lx0 + 0.20
    ax.add_patch(Circle((sxc, (gy0 + gy1) / 2), 0.065, fc="white", ec=IN, lw=0.7, zorder=5))
    c.text(sxc, (gy0 + gy1) / 2 - 0.004, "$s$", fs=FS + 0.5, color=IN, ha="center")
    c.text(sxc, yl1, "scale", fs=FS_S, color=IN, ha="center")
    # 744 l, drawn as the month's budget split into its 31 days
    ex0, ew = lx0 + 0.45, 0.13
    c.arrow((sxc + 0.07, (gy0 + gy1) / 2), (ex0 - 0.035, (gy0 + gy1) / 2), color=IN)
    cum = np.concatenate([[0], np.cumsum(p)])
    for i in range(31):
        fc = CAR_D if i == d_star else (CAR_L if i % 2 else "#C6E3DF")
        ax.add_patch(Rectangle((ex0, gy0 + gh * cum[i]), ew, gh * p[i], fc=fc, ec="white", lw=0.2, zorder=4))
    ax.add_patch(Rectangle((ex0, gy0), ew, gh, fc="none", ec=CARRIER, lw=0.6, zorder=5))
    ecx = ex0 + ew / 2
    c.text(ecx, yl1, r"$744\,\ell$", fs=FS_M, ha="center")
    c.text(lx0 + 0.07, yl2, r"$\ell=s\cdot\mathrm{softplus}(\cdot)$", fs=FS_S, color=GREY)
    # daily shares
    px0, pw = lx0 + 0.86, 0.92
    bw = pw / 31
    for i, v in enumerate(p):
        ax.add_patch(Rectangle((px0 + i * bw + 0.12 * bw, gy0), 0.76 * bw, gh * v / p.max(),
                               fc=CAR_D if i == d_star else CAR_M, ec="none", zorder=4))
    c.line([px0, px0 + pw], [gy0, gy0], color=CARRIER, lw=0.5, z=4)
    c.text(px0 + pw / 2, yl1, r"$p_d$", fs=FS_M, ha="center")
    c.text(px0 + pw / 2, yl2, "softmax over 31 days", fs=FS_S, color=GREY, ha="center")
    # hourly shares: one curve per day, the followed day dark, a weekend day light
    qx0, qw = lx0 + 2.02, 0.66
    qmax = q.max() * 1.05
    for i in range(31):
        if i != d_star and not work[i]:
            c.curve(qx0, gy0, qw, gh, q[i], CAR_L, lw=0.7, lo=0, hi=qmax, z=4)
    for i in range(31):
        if i != d_star and work[i]:
            c.curve(qx0, gy0, qw, gh, q[i], CAR_M, lw=0.35, lo=0, hi=qmax, z=4, alpha=0.6)
    c.curve(qx0, gy0, qw, gh, q[d_star], CAR_D, lw=1.0, lo=0, hi=qmax, z=5)
    c.line([qx0, qx0 + qw], [gy0, gy0], color=CARRIER, lw=0.5, z=4)
    c.text(qx0 + qw / 2, yl1, r"$q_{d,h}$", fs=FS_M, ha="center")
    c.text(qx0 + qw / 2, yl2, "rank-8 dictionary", fs=FS_S, color=GREY, ha="center")
    # their product
    fx0, fw = lx0 + 2.88, 0.94
    Fm = F.max() * 1.04
    ax.add_patch(Rectangle((fx0 + fw * d_star / 31, gy0), fw / 31, gh, fc=CAR_L, ec="none", alpha=0.7, zorder=3))
    c.curve(fx0, gy0, fw, gh, F.ravel(), CARRIER, lw=0.4, lo=0, hi=Fm, z=4)
    c.line([fx0, fx0 + fw], [gy0, gy0], color=CARRIER, lw=0.5, z=4)
    c.text(fx0 + fw / 2, yl1, r"$F_{d,h}$ (744 h)", fs=FS_M, ha="center")
    c.text(fx0 + fw / 2, yl2, r"mean of $F=\ell$ exactly", fs=FS_S, color=GREY, ha="center")
    for xo, sym in ((ex0 + ew + (px0 - ex0 - ew) / 2, r"$\times$"), (px0 + pw + (qx0 - px0 - pw) / 2, r"$\times$"),
                    (qx0 + qw + (fx0 - qx0 - qw) / 2, "=")):
        c.text(xo, (gy0 + gy1) / 2, sym, fs=9, color=GREY, ha="center")

    # ---- inputs of the decoder: copied daily means and within-day shares (pointer), z_d and v (read-out)
    for xa, lab in ((px0 + pw / 2, "+ daily means"), (qx0 + 0.16, "+ within-day shares")):
        c.arrow((xa, py1 + 0.004), (xa, dy0 - 0.003), color=HIST)
        c.text(xa + 0.045, (py1 + dy0) / 2, lab, fs=FS_S, color=HIST)
    xr = fx0 + fw / 2 + 0.06
    c.arrow((xr, ay1 + 0.004), (xr, dy0 - 0.003), color=CARRIER)
    c.text(xr + 0.045, (ay1 + dy0) / 2, r"$z_d,\ v$", fs=FS_S, color=CARRIER)
    if RELEASE == "0.2":
        _training_strip(c, lx0, rx1, dy1)


def _training_strip(c, lx0, rx1, below):
    """Release 0.2.0: how the carrier is trained (recipe Rw) and ensembled (six seeds), above the decoder of (b).
    Stage 1 from scratch with early stopping on the last 93 days before the cut; stage 2, 8 epochs with that span
    returned and the loss of each training window i weighted by w_i = max(sigma_i^2, floor)^(-1/2) (sigma_i^2: the
    week-lag noise of the window's own last three context weeks; floor: the 25% quantile over the pool; normalised to
    mean 1 within each source); six seeds, their forecasts averaged at every issue time."""
    ax = c.ax
    sy0, sy1 = below + 0.06, below + 0.62
    c.rbox(lx0, sy0, rx1 - lx0, sy1 - sy0, CAR_XL, CAR_L, lw=0.6, r=0.04, z=1, name="training")
    yt = sy1 - 0.09
    c.segments(lx0 + 0.07, yt, [("Training", CARRIER, {"fontweight": "bold"}),
                                ("   recipe Rw: span return with inverse-noise loss weights; six seeds", GREY, {})], fs=FS)
    # ---- stage timeline: 1 from scratch (93-day validation held out), 2 the span returned, weighted loss
    x0, x1, xv = lx0 + 0.25, lx0 + 1.62, lx0 + 1.27
    r1, r2, bh = sy1 - 0.27, sy1 - 0.42, 0.10                  # bottoms of the two stage bars
    for yy, k in ((r1, "1"), (r2, "2")):
        ax.add_patch(Circle((lx0 + 0.13, yy + bh / 2), 0.048, fc="white", ec=CARRIER, lw=0.6, zorder=5))
        c.text(lx0 + 0.13, yy + bh / 2 - 0.004, k, fs=FS_S, color=CARRIER, ha="center", fontweight="bold")
    c.rbox(x0, r1, xv - x0, bh, CAR_M, CAR_M, lw=0.4, r=0.015, z=4)
    ax.add_patch(Rectangle((xv, r1), x1 - xv, bh, fc="white", ec=CARRIER, lw=0.5, hatch="//////", zorder=4))
    c.text((x0 + xv) / 2, r1 + bh / 2, "from scratch, early stop", fs=FS_S, color="white", ha="center", z=8)
    c.text((xv + x1) / 2, r1 + bh / 2, "93 d", fs=FS_S, color=CAR_D, ha="center", z=8,
           bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
    c.rbox(x0, r2, x1 - x0, bh, CARRIER, CARRIER, lw=0.4, r=0.015, z=4)
    c.text((x0 + x1) / 2, r2 + bh / 2, "8 epochs, span returned", fs=FS_S, color="white", ha="center", z=8)
    rng = np.random.default_rng(17)                            # illustrative per-window loss weights under stage 2
    nb = 30
    s2 = np.exp(rng.normal(0.0, 0.8, nb))
    w = np.maximum(s2, np.quantile(s2, 0.25)) ** -0.5
    w = w / w.mean()
    wb0, wh = sy0 + 0.035, 0.085
    bw = (x1 - x0) / nb
    for i, v in enumerate(w):
        ax.add_patch(Rectangle((x0 + i * bw + 0.15 * bw, wb0), 0.7 * bw, wh * v / w.max(), fc=CAR_D, ec="none", zorder=4))
    c.text(lx0 + 0.13, wb0 + wh / 2, r"$w_i$", fs=FS_S, color=CAR_D, ha="center")
    # ---- the weight rule: w = max(sigma^2, floor)^(-1/2) on log axes (flat below the floor, slope -1/2 above)
    gx0, gx1, gy0, gy1 = lx0 + 1.84, lx0 + 2.30, sy0 + 0.10, sy1 - 0.17
    c.line([gx0, gx1], [gy0, gy0], color=GREY_M, lw=0.45, z=3)
    c.line([gx0, gx0], [gy0, gy1], color=GREY_M, lw=0.45, z=3)
    xf = gx0 + 0.34 * (gx1 - gx0)                              # the floor (25% quantile of sigma^2 over the pool)
    yhi = gy1 - 0.02
    c.line([gx0 + 0.01, xf, gx1 - 0.01], [yhi, yhi, yhi - 0.5 * (gx1 - 0.01 - xf)], color=CARRIER, lw=1.0, z=5)
    c.line([xf, xf], [gy0, yhi], color=GREY, lw=0.45, z=4, ls=(0, (1.5, 1.2)))
    c.text(gx0 - 0.03, yhi, r"$w$", fs=FS_S, color=CAR_D, ha="right")
    c.text(gx1, gy0 - 0.05, r"$\sigma^2$", fs=FS_S, color=GREY, ha="right")
    c.text(xf, gy0 - 0.05, "25%", fs=FS_S, color=GREY, ha="center")
    tx = gx1 + 0.07
    c.text(tx, sy1 - 0.215, r"$w_i=\max(\sigma_i^2,\,\sigma^2_{25\%})^{-1/2}$", fs=FS_S, color=INK)
    c.text(tx, sy1 - 0.325, "per training window $i$", fs=FS_S, color=GREY)
    c.text(tx, sy1 - 0.435, r"$\sigma_i^2$: its context noise", fs=FS_S, color=GREY)
    # ---- six seeds, their forecasts averaged at every issue time
    kw, kh, dk = 0.17, 0.12, 0.022
    kx, ky = rx1 - 0.44, sy0 + 0.17
    for j in range(6):
        c.rbox(kx + dk * j, ky + 0.014 * j, kw, kh, CARRIER if j == 0 else CAR_M, "white", lw=0.4, r=0.015,
               z=4 + 0.1 * (6 - j))                            # seed cards stacked back to the upper right; the front one labelled
    c.text(kx + kw / 2, ky + kh / 2, "×6", fs=FS_S, color="white", ha="center", z=8, fontweight="bold")
    c.text(kx + (kw + 5 * dk) / 2, sy0 + 0.085, "seeds averaged", fs=FS_S, color=CARRIER, ha="center")


# ---------------------------------------------------------------------------- panel (c): the anchoring layer
def _panel_c(c, cost_text):
    ax = c.ax
    x0, y0, x1, y1 = PC
    c.rbox(x0, y0, x1 - x0, y1 - y0, "white", FRAME, lw=0.5, r=0.06, z=0, name="panel c")
    cont = RELEASE == "0.2"
    if cont:
        c.letter(x0 + 0.08, y1 - 0.13, "c", "Anchoring and continuation layers", "nothing trained")
    else:
        c.letter(x0 + 0.08, y1 - 0.13, "c", "Anchoring layer", "copy of ANKYRA 2.2.1; nothing trained")

    bx0, bx1 = x0 + 0.10, x1 - 0.10            # row boxes
    gx0, gw = bx0 + 0.14, 0.86                 # glyph column
    tx = gx0 + gw + 0.12                       # text column
    rh, gap = 0.64, (0.10 if cont else 0.12)
    tops = [y1 - 0.26 - i * (rh + gap) for i in range(3)]
    names = ("level", "daily means", "within-day")
    for top, nm in zip(tops, names):
        c.rbox(bx0, top - rh, bx1 - bx0, rh, LITE_XL, LITE_L, lw=0.5, r=0.04, z=1, name=nm)
    lh = 0.118                                 # line pitch
    l1 = lambda top: top - 0.10                # noqa: E731  title line
    tagx = bx1 - 0.07

    # ---- level: six historical candidates + the carrier's window mean, inverse-MSE weights shrunk to equal
    top = tops[0]
    c.text(tx, l1(top), "Level", fs=FS, fontweight="bold", color=LITE)
    c.tag(tagx, l1(top), "$e_k,\\ k\\leq6$")
    c.text(tx, l1(top) - lh - 0.01, r"$\ell^{A}=\Sigma_c\,\pi_c\,\ell_c$,  $\pi_c\propto1/\mathrm{MSE}_c$", fs=FS_M)
    c.segments(tx, l1(top) - 2 * lh - 0.02, [("6 historical candidates", HIST, {}), (" + carrier ", CARRIER, {}),
                                             (r"$\ell$", CARRIER, {})], fs=FS_S)
    c.text(tx, l1(top) - 3 * lh - 0.02, r"shrunk to equal weights ($K_0=8$)", fs=FS_S, color=GREY)
    wts = np.array([0.10, 0.16, 0.07, 0.19, 0.09, 0.11, 0.28])
    base, hmax = top - rh + 0.11, 0.40
    bw = gw / 7
    for i, v in enumerate(wts):
        ax.add_patch(Rectangle((gx0 + i * bw + 0.18 * bw, base), 0.64 * bw, hmax * v / wts.max(),
                               fc=CARRIER if i == 6 else HIST, ec="none", zorder=4))
    c.line([gx0, gx0 + gw], [base, base], color=GREY_M, lw=0.5, z=4)
    c.line([gx0, gx0 + gw], [base + hmax * (1 / 7) / wts.max()] * 2, color=GREY, lw=0.5, z=5, ls=(0, (2, 1.5)))
    c.text(gx0 + 3 * bw, base - 0.055, "history", fs=FS_S, color=HIST, ha="center")
    c.text(gx0 + 6.5 * bw, base - 0.055, "$\\ell$", fs=FS_S, color=CARRIER, ha="center")
    c.text(gx0 + 0.01, base + hmax + 0.005, r"$\pi_c$", fs=FS_S, color=LITE, va="top")

    # ---- daily means: history side handed over to the carrier's daily means, one weight per window of days
    top = tops[1]
    c.text(tx, l1(top), "Daily means", fs=FS, fontweight="bold", color=LITE)
    c.tag(tagx, l1(top), "$e_k,\\ k\\leq6$")
    c.text(tx, l1(top) - lh - 0.01, r"$m_d=(1-\alpha_w)\,m^{H}_d+\alpha_w\,m^{C}_d$", fs=FS_M)
    c.segments(tx, l1(top) - 2 * lh - 0.02, [(r"$m^{H}$: history", HIST, {}), (",  ", GREY, {}),
                                             (r"$m^{C}_d=31\,\ell\,p_d$", CARRIER, {})], fs=FS_S)
    c.text(tx, l1(top) - 3 * lh - 0.02, r"$\alpha_w$ shrunk to ½ ($K_0=2$)", fs=FS_S, color=GREY)
    d = np.arange(31)
    wk = np.where(((d + 3) % 7) < 5, 1.0, 0.55)
    mH = 0.93 * wk + 0.02 * np.sin(d / 3)
    mC = 1.05 * wk - 0.03 * np.cos(d / 4)
    al = [0.74, 0.61, 0.49, 0.38]
    windows = ((0, 7), (7, 14), (14, 21), (21, 31))
    alr = np.concatenate([[a] * (w1 - w0) for (w0, w1), a in zip(windows, al)])
    m = (1 - alr) * mH + alr * mC
    cy0, ch = top - rh + 0.12, 0.25           # curves
    aby, abh = cy0 + ch + 0.035, 0.20          # alpha bars: baseline, height of alpha = 1
    for (w0, w1), a in zip(windows, al):
        xa0, xa1 = gx0 + gw * w0 / 31, gx0 + gw * w1 / 31
        ax.add_patch(Rectangle((xa0 + 0.010, aby), xa1 - xa0 - 0.020, abh * a, fc=LITE_L, ec=LITE, lw=0.5,
                               zorder=3))
        if w0:
            c.line([xa0, xa0], [cy0 - 0.02, aby + abh * 0.8], color=GREY_M, lw=0.35, z=2, ls=(0, (1, 1)))
    c.line([gx0, gx0 + gw], [aby + abh * 0.5] * 2, color=GREY, lw=0.5, z=5, ls=(0, (2, 1.5)))
    c.text(gx0 - 0.015, aby + abh * 0.5, "½", fs=FS_S, color=GREY, ha="right")
    c.text(gx0 - 0.005, aby + abh * 0.95, r"$\alpha_w$", fs=FS_S, color=LITE, ha="right")
    c.curve(gx0, cy0, gw, ch, mH, HIST, lw=0.7, lo=0.45, hi=1.12)
    c.curve(gx0, cy0, gw, ch, mC, CARRIER, lw=0.7, lo=0.45, hi=1.12)
    c.curve(gx0, cy0, gw, ch, m, LITE, lw=1.0, lo=0.45, hi=1.12, z=6)
    for dd in (1, 8, 15, 22, 31):
        c.text(gx0 + gw * (dd - 0.5) / 31, cy0 - 0.06, str(dd), fs=FS_S, color=GREY, ha="center")

    # ---- within-day shape: the carrier's shape moved towards the unit's analog-day shape by one trust omega
    top = tops[2]
    c.text(tx, l1(top), "Within-day shape", fs=FS, fontweight="bold", color=LITE)
    c.tag(tagx, l1(top), "$e_k,\\ k\\leq3$")
    c.text(tx, l1(top) - lh - 0.01, r"$w=w^{C}+\omega\,(S-w^{C})$", fs=FS_M)
    c.segments(tx, l1(top) - 2 * lh - 0.02, [(r"$S$: analog-day shape", HIST, {}), (",  ", GREY, {}),
                                             (r"$w^{C}=F-m^{C}$", CARRIER, {})], fs=FS_S)
    yg = l1(top) - 3 * lh - 0.02                # omega gauge: one value per window, capped at 1/2
    om = 0.31
    c.text(tx, yg, r"$\omega$", fs=FS, color=LITE)
    g0, g1 = tx + 0.13, tx + 0.78
    ax.add_patch(Rectangle((g0, yg - 0.027), g1 - g0, 0.054, fc="white", ec=GREY_M, lw=0.45, zorder=4))
    ax.add_patch(Rectangle((g0, yg - 0.027), (g1 - g0) * om, 0.054, fc=LITE, ec="none", alpha=0.85, zorder=5))
    ax.add_patch(Rectangle(((g0 + g1) / 2, yg - 0.027), (g1 - g0) / 2, 0.054, fc="none", ec="#A8A8A8", lw=0,
                           hatch="//////", zorder=5))      # hatch in the edge colour (Matplotlib >= 3.8); no edge drawn
    c.line([(g0 + g1) / 2] * 2, [yg - 0.05, yg + 0.05], color=INK, lw=0.6, z=6)
    c.text(g1 + 0.06, yg, r"$\omega\leq$ ½", fs=FS_S, color=GREY)
    hh = np.arange(24) + 0.5
    wc = _office_day(hh, True, 0.8)
    s_ = _office_day(hh - 0.8, True, 0.95)
    w = wc + om * (s_ - wc)
    cy0, ch = top - rh + 0.07, 0.45
    c.curve(gx0, cy0, gw, ch, s_, HIST, lw=0.7, lo=0.3, hi=1.45)
    c.curve(gx0, cy0, gw, ch, wc, CARRIER, lw=0.7, lo=0.3, hi=1.45)
    c.curve(gx0, cy0, gw, ch, w, LITE, lw=1.0, lo=0.3, hi=1.45, z=6)
    c.text(gx0 + 0.005, cy0 + ch - 0.03, "$S$", fs=FS_S, color=HIST)
    c.text(gx0 + 0.005, cy0 + ch - 0.15, "$w^{C}$", fs=FS_S, color=CARRIER)
    c.text(gx0 + gw - 0.005, cy0 + ch - 0.03, "24 h", fs=FS_S, color=GREY, ha="right")

    # ---- flow between the rows: level -> daily means; daily means + within-day -> output
    xg = gx0 + gw / 2
    c.arrow((xg, tops[0] - rh - 0.004), (xg, tops[1] + 0.003), color=LITE)
    c.text(xg, (tops[1] - rh + tops[2]) / 2, "+", fs=FS + 2, color=LITE, ha="center", fontweight="bold")
    oy1 = tops[2] - rh - gap
    oy0 = y0 + 0.08
    if cont:
        ktop = oy1
        oy1 = _continuation_row(c, bx0, bx1, gx0, gw, tx, tagx, ktop, lh) - gap
        c.arrow((xg, tops[2] - rh - 0.004), (xg, ktop + 0.003), color=LITE)
        c.text(xg + 0.06, (tops[2] - rh + ktop) / 2, r"$F_{d,h}=\max(m_d+w_{d,h},\,0)$, then the state rule", fs=FS_S, color=GREY)
        xk = gx0 + gw / 2
        c.arrow((xk, oy1 + gap - 0.004), (xk, oy1 + 0.003), color=LITE)
    else:
        c.arrow((xg, tops[2] - rh - 0.004), (xg, oy1 + 0.003), color=LITE)

    # ---- output
    c.rbox(bx0, oy0, bx1 - bx0, oy1 - oy0, "white", LITE, lw=0.9, r=0.04, z=1, name="output")
    if cont:
        c.segments(bx0 + 0.08, oy1 - 0.10, [("ankyra-lite", LITE, {"fontweight": "bold"}), ("   ", INK, {}),
                                            (r"delivered $F'_{d,h}$, 744 h", INK, {})], fs=FS)
        lines = [r"energy $744\,\bar m$, peak $\mathrm{max}_d\,(\hat L_d+A_{\tau_d})$:", "  read from $F$, before the continuation layer",
                 "state windows: seasonal naive (week), no layer"]
    else:
        c.segments(bx0 + 0.08, oy1 - 0.10, [("ankyra-lite", LITE, {"fontweight": "bold"}), ("   ", INK, {}),
                                            (r"$\hat F_{d,h}=\max(m_d+w_{d,h},\,0)$, 744 h", INK, {})], fs=FS)
        lines = [r"energy $744\,\bar m$;   peak $\mathrm{max}_d\,(\hat L_d+A_{\tau_d})$",
                 "state windows: weekly seasonal naive"]
    if cost_text:
        lines.append(" ".join(str(cost_text).split()))
    for i, s in enumerate(lines):
        c.text(bx0 + 0.08, oy1 - 0.235 - i * 0.112, s, fs=FS_S, color=GREY)
    # the delivered trajectory (first week), schematic
    hh = np.arange(24 * 7) % 24 + 0.5
    wday = (np.arange(24 * 7) // 24 + 3) % 7 < 5
    tr = _office_day(hh, wday, 0.8)
    sx0, sx1, sy0, sy1 = bx1 - 0.66, bx1 - 0.08, oy0 + 0.07, oy0 + 0.30
    xs = np.linspace(sx0, sx1, len(tr))
    ys = sy0 + (sy1 - sy0) * tr / tr.max()
    ax.fill_between(xs, sy0, ys, color=LITE_XL, lw=0, zorder=2)
    ax.plot(xs, ys, color=LITE, lw=0.6, zorder=3)
    c.line([sx0, sx1], [sy0, sy0], color=GREY_M, lw=0.4, z=3)


def _continuation_row(c, bx0, bx1, gx0, gw, tx, tagx, top, lh, rh=0.58):
    """Release 0.2.0: the origin-continuity layer (ankyra_lite/continuation.py) as the last row of panel (c); returns
    the row's bottom.  F'(h) = F(h) exp(delta(h)), delta(h) = kappa1 x_f exp(-h / 12) + kappa2 x_d exp(-h / 168) with
    kappa1 = 0.479, kappa2 = 0.330; x_f and x_d are the clipped log ratios of the last 3 and 24 observed hours to the same
    hours of the reference forecast day.  The glyph is synthetic: the last day before the origin above the forecast, F'
    pulled toward it in the first hours and back to F within the first week."""
    ax = c.ax
    c.rbox(bx0, top - rh, bx1 - bx0, rh, "white", LITE, lw=0.6, r=0.04, z=1, name="continuation")
    l1 = top - 0.10
    c.text(tx, l1, "Continuation", fs=FS, fontweight="bold", color=LITE)
    c.tag(tagx, l1, "last 24 h before $o$")
    c.text(tx, l1 - lh - 0.01, r"$F'(h)=F(h)\,e^{\delta(h)}$,  $h=0\ldots743$", fs=FS_M)
    c.text(tx, l1 - 2 * lh - 0.02, r"$\delta(h)=\kappa_1x_f\,e^{-h/12}+\kappa_2x_d\,e^{-h/168}$", fs=FS_S)
    c.text(tx, l1 - 3 * lh - 0.02, r"$x_f,x_d$: last 3 h, 24 h vs $F$ (log, $\pm1$)", fs=FS_S, color=GREY)
    # glyph: 24 h before and 72 h after the origin
    hb = np.arange(-24, 0) + 0.5
    hf = np.arange(0, 72) + 0.5
    obs = _office_day((hb + 24) % 24, True, 0.95) * 1.16 + 0.03 * np.sin(hb / 2.3)
    F = _office_day(hf % 24, True, 0.95)
    xf_, xd_ = 0.42, 0.20                                          # illustrative anomalies (clipped log ratios)
    Fp = F * np.exp(0.479 * xf_ * np.exp(-hf / 12) + 0.330 * xd_ * np.exp(-hf / 168))
    g0, g1 = gx0, gx0 + gw
    xo = g0 + gw * 24 / 96
    X_ = lambda h: g0 + gw * (np.asarray(h) + 24) / 96             # noqa: E731
    cy0, ch = top - rh + 0.10, 0.36
    lo, hi = 0.3, 1.75
    Y_ = lambda v: cy0 + ch * (np.asarray(v) - lo) / (hi - lo)     # noqa: E731
    ax.plot(X_(hb), Y_(obs), color=OBS_D, lw=0.7, zorder=5)
    ax.plot(X_(hf), Y_(F), color=LITE_L, lw=0.8, zorder=5, ls=(0, (2.2, 1.2)))
    ax.plot(X_(hf), Y_(Fp), color=LITE, lw=1.0, zorder=6)
    c.line([g0, g1], [cy0, cy0], color=GREY_M, lw=0.4, z=3)
    c.line([xo, xo], [cy0, cy0 + ch + 0.02], color=GREY, lw=0.5, z=4, ls=(0, (1.5, 1.2)))
    c.text(xo, cy0 - 0.055, "$o$", fs=FS_S, color=GREY, ha="center")
    c.text(g1, cy0 - 0.055, "+72 h", fs=FS_S, color=GREY, ha="right")
    c.text(g0, cy0 - 0.055, "−24", fs=FS_S, color=GREY, ha="left")
    c.text(X_(30), Y_(1.62), "$F'$", fs=FS_S, color=LITE, ha="left")
    c.text(X_(58), Y_(0.30) + 0.05, "$F$", fs=FS_S, color=LITE_L, ha="left")
    return top - rh


if __name__ == "__main__":
    # python -B architecture.py <path_stem> ["cost line"]   (a preview; the published Figure 1 is drawn by
    # make_figures.py, which reads the parameter count and the time per forecast from results/cost_per_window.csv)
    if len(sys.argv) < 2:
        sys.exit('usage: python -B architecture.py <path_stem> ["time per forecast line"]')
    out = Path(sys.argv[1])
    problems = draw(out, cost_text=sys.argv[2] if len(sys.argv) > 2 else None)
    print("wrote", str(out) + ".pdf", str(out) + ".png", f"({WIDTH} x {HEIGHT:.2f} in)")
    for p_ in problems:
        print("  ", p_.encode("ascii", "backslashreplace").decode("ascii"))
    print("  layout problems:", len(problems))
