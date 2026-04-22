"""Build MNRAS-formatted PDF from docs/paper/paper_draft.md.

Pipeline:
  1. Preprocess markdown — rewrite `../../notebooks/figures/X.png` paths
     to bare filenames and strip pandoc attribute blocks pandoc doesn't
     accept from our markdown dialect.
  2. Copy all referenced figures into docs/paper/build/figures/ so the
     LaTeX graphicspath is a stable sibling of the .tex.
  3. `pandoc --to latex` → paper_body.tex (LaTeX fragment, no preamble).
  4. Write paper_final.tex wrapper with \\documentclass{mnras}, title,
     author, abstract.
  5. Compile with tectonic, placing the PDF at docs/paper/paper_final.pdf.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MD_SRC = ROOT / "docs" / "paper" / "paper_draft.md"
BUILD = ROOT / "docs" / "paper" / "build"
FIG_SRC = ROOT / "notebooks" / "figures"
# v2 adds geometry + \resizebox around tables + \linewidth on all
# figures + \sloppy to fix right-edge clipping seen in v1.
FINAL_PDF = ROOT / "docs" / "paper" / "paper_final_v2.pdf"


# Abstract extracted verbatim for the \begin{abstract} block (since MNRAS
# wants it outside the body). Must match the markdown source.
ABSTRACT_TEXT = (
    "We present Taylor-CNN, a physics-informed 1D CNN for exoplanet transit "
    "classification on phase-folded light curves. The model replaces a single "
    "learned planet-shape filter with a bank of five fixed-morphology gates "
    "(planet U, V, inverted secondary, asymmetric ingress, narrow Gaussian), "
    "each with a learnable amplitude. A dataset-calibrated geometry-consistency "
    "loss regularises training toward known class-separation directions. On a "
    "76-TCE stratified Kepler DR25 test set the model (V10) achieves "
    "F1 = 0.861 / precision = 82.9\\% / recall = 89.5\\% with 1150 parameters. "
    "A V6 + V10 ensemble lifts F1 to 0.872 (precision 85.0\\%) and a three-model "
    "OR ensemble reaches 100\\% recall. Zero-shot transfer to TESS achieves "
    "100\\% recall without retraining, compared to 63\\% for statistical-feature "
    "approaches (Malik et al. 2022)."
)


MNRAS_WRAPPER = r"""\documentclass[referee,usenatbib]{mnras}

% MNRAS already loads \usepackage{geometry}; override the margins with
% \geometry{...} so tables and figures have more horizontal room.
\geometry{left=2cm,right=2cm,top=2.5cm,bottom=2.5cm}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{calc}
\usepackage{array}

% Give LaTeX more flexibility to break lines so long \texttt / URLs / table
% cells don't spill past the right margin (the v1 right-edge clipping).
\sloppy
\emergencystretch=3em
\tolerance=2000
\hbadness=10000

\graphicspath{{figures/}}

% Pandoc helper macros (pandoc's fragment output assumes these exist; our
% MNRAS preamble doesn't otherwise define them).
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\pandocbounded}[1]{#1}
\providecommand{\real}[1]{#1}
\providecommand{\CSLReferences}[2]{}

\title[Taylor-CNN Transit Classifier]{Taylor-CNN: Physics-Informed Transit
Classification with a Multi-Template Gate Bank}

\author[S.\ Kapali]{%
Srikanth Kapali$^{1}$\thanks{E-mail: srikanth.kapali@gmail.com}
\\
$^{1}$Independent research -- personal ML/AI learning project.
}

\date{__DATE__}

\pubyear{2026}

\begin{document}
\label{firstpage}

\maketitle

\begin{abstract}
__ABSTRACT__
\end{abstract}

\begin{keywords}
methods: data analysis -- techniques: photometric -- planets and satellites:
detection
\end{keywords}

__BODY__

\bsp
\label{lastpage}
\end{document}
"""


def log(msg):
    print(msg, flush=True)


def preprocess_markdown() -> str:
    """Load the draft and rewrite image paths + strip attribute blocks."""
    md = MD_SRC.read_text(encoding="utf-8")

    # Rewrite ![caption](../../notebooks/figures/X.png){...}  ->
    # ![caption](X.png)   (drop the attribute block — pandoc's raw_attribute
    # extension is touchy; we rely on graphicspath to resolve the basename.)
    md = re.sub(
        r"!\[([^\]]*)\]\(\.\./\.\./notebooks/figures/([^)]+)\)\{[^}]*\}",
        r"![\1](\2)",
        md,
    )
    # Any remaining {...} attribute blocks on images
    md = re.sub(
        r"(!\[[^\]]*\]\([^)]+\))\{[^}]*\}",
        r"\1",
        md,
    )

    # Strip the abstract block — we emit it in the MNRAS \begin{abstract}.
    md = re.sub(
        r"## Abstract\s+.*?(?=^## 1\.)",
        "",
        md,
        flags=re.DOTALL | re.MULTILINE,
    )

    # Also drop the "*Draft paper*" italic note under the title.
    md = re.sub(r"\*Draft paper[^*]+\*\n+", "", md, count=1)

    # Drop the top-level H1 (title handled by MNRAS).
    md = re.sub(r"^# [^\n]+\n+", "", md, count=1, flags=re.MULTILINE)

    return md


def list_figures(md: str) -> list[str]:
    """Return the unique PNG basenames the markdown references."""
    return sorted({m for m in re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", md)})


def prepare_build_dir(md: str) -> tuple[Path, list[str]]:
    BUILD.mkdir(parents=True, exist_ok=True)
    fig_dst = BUILD / "figures"
    fig_dst.mkdir(exist_ok=True)

    figures = list_figures(md)
    missing = []
    for fig in figures:
        src = FIG_SRC / fig
        if not src.exists():
            missing.append(fig); continue
        shutil.copy2(src, fig_dst / fig)
    if missing:
        log(f"  WARNING: {len(missing)} figures missing: {missing}")

    md_path = BUILD / "paper_body.md"
    md_path.write_text(md, encoding="utf-8")
    return md_path, figures


def run_pandoc(md_path: Path) -> Path:
    body_tex = BUILD / "paper_body.tex"
    cmd = [
        "pandoc",
        str(md_path),
        "--to", "latex",
        "--wrap=preserve",
        "-o", str(body_tex),
    ]
    log(f"  pandoc: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return body_tex


_UNICODE_TO_TEX = {
    "λ": r"\ensuremath{\lambda}",
    "π": r"\ensuremath{\pi}",
    "σ": r"\ensuremath{\sigma}",
    "ρ": r"\ensuremath{\rho}",
    "μ": r"\ensuremath{\mu}",
    "Δ": r"\ensuremath{\Delta}",
    "α": r"\ensuremath{\alpha}",
    "β": r"\ensuremath{\beta}",
    "×": r"\ensuremath{\times}",
    "·": r"\ensuremath{\cdot}",
    "−": r"\ensuremath{-}",
    "²": r"\ensuremath{^{2}}",
    "³": r"\ensuremath{^{3}}",
    "≥": r"\ensuremath{\geq}",
    "≤": r"\ensuremath{\leq}",
    "≠": r"\ensuremath{\neq}",
    "→": r"\ensuremath{\rightarrow}",
    "←": r"\ensuremath{\leftarrow}",
    "∈": r"\ensuremath{\in}",
    "∞": r"\ensuremath{\infty}",
    "≈": r"\ensuremath{\approx}",
    "…": r"\ldots{}",
    "—": r"---",
    "–": r"--",
    " ": r"~",
    "−": r"\ensuremath{-}",  # U+2212 MINUS SIGN
    "±": r"\ensuremath{\pm}",
    "°": r"\ensuremath{^{\circ}}",
    "∂": r"\ensuremath{\partial}",
    "√": r"\ensuremath{\surd}",
    "⋅": r"\ensuremath{\cdot}",
    "∇": r"\ensuremath{\nabla}",
    "∑": r"\ensuremath{\sum}",
    "∫": r"\ensuremath{\int}",
    " ": r"\,",  # thin space
    " ": r" ",   # en space
    "—": r"---",  # em dash
    "–": r"--",   # en dash
    "‘": r"`",    # left single quote
    "’": r"'",    # right single quote
    "“": r"``",   # left double quote
    "”": r"''",   # right double quote
}


_VERBATIM_ASCII = {
    "λ": "lambda",
    "π": "pi",
    "σ": "sigma",
    "ρ": "rho",
    "μ": "mu",
    "Δ": "Delta",
    "α": "alpha",
    "β": "beta",
    "×": "x",
    "·": "*",
    "−": "-",
    "²": "^2",
    "³": "^3",
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    "→": "->",
    "←": "<-",
    "∈": "in",
    "∞": "inf",
    "≈": "~",
    "∂": "d",
    "…": "...",
    "—": "--",
    "–": "-",
}


def _transliterate_unicode(tex: str) -> str:
    """Replace Unicode in regular text with \\ensuremath{\\lambda} etc., but
    inside verbatim environments use ASCII equivalents because verbatim
    doesn't interpret LaTeX macros — `\\ensuremath` would render literally
    AND the expanded form (21 chars vs 1) overflows the line."""
    verb_pat = re.compile(
        r"(\\begin\{verbatim\}.*?\\end\{verbatim\})", re.DOTALL
    )
    parts = verb_pat.split(tex)
    rebuilt = []
    for i, chunk in enumerate(parts):
        if i % 2 == 1:
            # verbatim block: use ASCII-only replacements, wrap in footnotesize
            for src, dst in _VERBATIM_ASCII.items():
                chunk = chunk.replace(src, dst)
            chunk = "\\begingroup\\footnotesize\n" + chunk + "\n\\endgroup"
        else:
            for src, dst in _UNICODE_TO_TEX.items():
                chunk = chunk.replace(src, dst)
        rebuilt.append(chunk)
    return "".join(rebuilt)


def _fit_includegraphics(tex: str) -> str:
    """Force every \\includegraphics to width=\\linewidth so images never
    spill past the right margin. Preserves keepaspectratio and alt= if
    pandoc emitted them."""
    pattern = re.compile(r"\\includegraphics\[([^\]]*)\]")
    def _rewrite(m):
        opts = m.group(1)
        if "width=" in opts:
            return m.group(0)
        opts = "width=\\linewidth," + opts
        return f"\\includegraphics[{opts}]"
    return pattern.sub(_rewrite, tex)


def _wrap_longtables_resizebox(tex: str) -> str:
    """Wrap every pandoc longtable in \\resizebox so wide tables shrink to
    fit \\linewidth. longtable supports paging across pages but not scale-
    to-fit, so we convert each to a plain tabular inside a table float.
    """
    start = 0
    out = []
    while True:
        s = tex.find(r"\begin{longtable}", start)
        if s == -1:
            out.append(tex[start:])
            break
        e = tex.find(r"\end{longtable}", s)
        if e == -1:
            out.append(tex[start:])
            break
        e += len(r"\end{longtable}")
        block = tex[s:e]

        tab = block
        # Strip longtable-only commands.
        for rm in (
            r"\endfirsthead",
            r"\endhead",
            r"\endfoot",
            r"\endlastfoot",
            r"\noalign{}",
        ):
            tab = tab.replace(rm, "")
        # Rename environment.
        tab = tab.replace(r"\begin{longtable}", r"\begin{tabular}")
        tab = tab.replace(r"\end{longtable}", r"\end{tabular}")
        # pandoc emits column widths as p{(\linewidth - N\tabcolsep) * \real{0.XX}} —
        # inside a \resizebox this can misbehave; keep as-is, the resizebox will
        # scale the final box either way.

        wrapped = (
            "\\begin{table}[ht!]\n"
            "\\centering\\footnotesize\n"
            "\\resizebox{\\linewidth}{!}{%\n"
            + tab +
            "\n}\n"
            "\\end{table}\n"
        )
        out.append(tex[start:s])
        out.append(wrapped)
        start = e
    return "".join(out)


def write_final_tex(body_tex: Path) -> Path:
    body = body_tex.read_text(encoding="utf-8")
    # Unicode -> LaTeX macros (MNRAS default font lacks Greek glyphs).
    body = _transliterate_unicode(body)
    # Force images to \linewidth.
    body = _fit_includegraphics(body)
    # Shrink wide tables to fit.
    body = _wrap_longtables_resizebox(body)
    # Force figure placement near the prose.
    body = body.replace(r"\begin{figure}", r"\begin{figure}[ht!]")

    from datetime import date
    wrapper = (MNRAS_WRAPPER
               .replace("__DATE__", date.today().isoformat())
               .replace("__ABSTRACT__", _transliterate_unicode(ABSTRACT_TEXT))
               .replace("__BODY__", body))
    final_tex = BUILD / "paper_final.tex"
    final_tex.write_text(wrapper, encoding="utf-8")
    return final_tex


def run_tectonic(final_tex: Path) -> Path:
    cmd = ["tectonic", "--keep-intermediates",
           "--outdir", str(BUILD), str(final_tex)]
    log(f"  tectonic: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    out = BUILD / "paper_final.pdf"
    if not out.exists():
        raise RuntimeError(f"tectonic did not produce {out}")
    return out


def main():
    log(f"Building from {MD_SRC}")
    md = preprocess_markdown()
    md_path, figs = prepare_build_dir(md)
    log(f"  copied {len(figs)} figures into {BUILD / 'figures'}")
    body_tex = run_pandoc(md_path)
    log(f"  wrote {body_tex}")
    final_tex = write_final_tex(body_tex)
    log(f"  wrote {final_tex}")
    pdf = run_tectonic(final_tex)
    log(f"  produced {pdf} ({pdf.stat().st_size / 1024:.1f} KB)")

    FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, FINAL_PDF)
    log(f"\nFinal PDF: {FINAL_PDF} ({FINAL_PDF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    sys.exit(main())
