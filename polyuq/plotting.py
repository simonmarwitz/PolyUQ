"""
polyuq.plotting — matplotlib style helpers for publication figures.

The `get_pcd` function returns a dict of rcParams suitable for print or
presentation contexts. LaTeX rendering is enabled only when a working
LaTeX + lualatex installation is detected; otherwise the function falls
back to the default matplotlib font renderer so that notebooks run on
any machine.
"""
import shutil
import matplotlib


def _latex_available() -> bool:
    return shutil.which("lualatex") is not None


def get_pcd(purpose: str = "print") -> dict:
    """Return an rcParams dict for the given *purpose*.

    Parameters
    ----------
    purpose : {"print", "print_half", "print_half_hor", "beamer", "beamer_half"}
        ``"print"`` targets a 150 mm column width (thesis/journal figures).
        ``"beamer"`` targets a half-slide width.

    Returns
    -------
    dict
        Pass to ``matplotlib.rc_context(get_pcd(...))`` as a context manager.
    """
    col_width = 5.906          # inches, 150 mm
    golden = 1.618

    size_map = {
        "print":          (col_width,          col_width / golden),
        "print_half":     (col_width / 2,      col_width / golden),
        "print_half_hor": (col_width,           col_width / golden / 2),
        "beamer":         (5.53,               2.96),
        "beamer_half":    (5.53 / 2,           2.96),
    }
    figsize = size_map.get(purpose, size_map["print"])
    font_size = 9 if "half" in purpose else 10

    pcd: dict = {
        "figure.figsize":       figsize,
        "figure.dpi":           100,
        "font.size":            font_size,
        "legend.fontsize":      font_size,
        "axes.labelsize":       font_size,
        "xtick.labelsize":      font_size,
        "ytick.labelsize":      font_size,
        "font.family":          "serif",
        "legend.labelspacing":  0.1,
        "axes.linewidth":       0.5,
        "xtick.major.width":    0.5,
        "ytick.major.width":    0.5,
    }

    if _latex_available():
        pcd.update({
            "text.usetex":      True,
            "pgf.texsystem":    "lualatex",
            "pgf.rcfonts":      False,
            "pgf.preamble":     (
                r"\usepackage{siunitx}\usepackage{xfrac}\usepackage{amsmath}"
                r"\usepackage{unicode-math}"
                r"\setmathfont{latinmodern-math.otf}"
                r"\setmathfont[range=\mathfrak]{Old English Text MT}"
                r"\setmathfont[range=\mathbb]{texgyrepagella-math.otf}"
            ),
        })
    else:
        # Graceful fallback: no LaTeX required.
        pcd["text.usetex"] = False

    return pcd
