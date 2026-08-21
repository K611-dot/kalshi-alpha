"""Single-file HTML reports.

Everything -- CSS, charts, tables -- is inlined into one file. A research
artefact that depends on a CDN or a sibling ``assets/`` directory stops
rendering the moment it is emailed, archived, or opened six months later, and a
result you cannot reopen is a result you did not produce.

Charts render through matplotlib when it is installed and degrade to tables
when it is not, so the report is never a hard dependency of the analysis.
"""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CSS = """
:root {
  --bg: #ffffff; --fg: #111418; --muted: #5b6570; --line: #e3e7eb;
  --accent: #1f6feb; --pos: #12805c; --neg: #b3261e; --card: #f7f9fb;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1216; --fg:#e6edf3; --muted:#9aa6b2; --line:#232a31;
          --accent:#589bff; --pos:#3fb950; --neg:#f85149; --card:#161b22; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
       font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1080px; margin-inline:auto; }
h1 { font-size:1.9rem; margin:0 0 .25rem; letter-spacing:-.02em; }
h2 { font-size:1.25rem; margin:2.5rem 0 .5rem; padding-bottom:.35rem;
     border-bottom:1px solid var(--line); letter-spacing:-.01em; }
h3 { font-size:1rem; margin:1.5rem 0 .35rem; color:var(--muted);
     text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
p, li { color:var(--fg); }
.sub { color:var(--muted); margin:0 0 2rem; }
.note { color:var(--muted); font-size:.9rem; }
pre { background:var(--card); border:1px solid var(--line); border-radius:8px;
      padding:.9rem 1rem; overflow-x:auto; font-family:var(--mono); font-size:12.5px;
      line-height:1.5; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:.5rem 0 1rem;
        display:block; overflow-x:auto; }
th, td { text-align:right; padding:.4rem .6rem; border-bottom:1px solid var(--line);
         white-space:nowrap; }
th { color:var(--muted); font-weight:600; text-transform:uppercase;
     font-size:11px; letter-spacing:.05em; }
td:first-child, th:first-child { text-align:left; }
tbody tr:hover { background:var(--card); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
         gap:.75rem; margin:1rem 0 1.5rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:.85rem 1rem; }
.card .k { color:var(--muted); font-size:11px; text-transform:uppercase;
           letter-spacing:.06em; }
.card .v { font-size:1.35rem; font-weight:650; font-variant-numeric:tabular-nums;
           margin-top:.15rem; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
img { max-width:100%; height:auto; display:block; margin:1rem 0;
      border:1px solid var(--line); border-radius:8px; background:#fff; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line);
         color:var(--muted); font-size:12.5px; }
"""


def esc(text: object) -> str:
    return html.escape(str(text))


def frame_to_html(df: pd.DataFrame, max_rows: int = 60, float_fmt: str = "{:.4g}") -> str:
    """Render a DataFrame as a compact table, truncating long ones honestly."""
    if df is None or df.empty:
        return '<p class="note">No rows.</p>'
    shown = df.head(max_rows)
    head = "".join(f"<th>{esc(c)}</th>" for c in shown.columns)
    rows = []
    for _, row in shown.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"<td>{esc(float_fmt.format(value))}</td>")
            else:
                cells.append(f"<td>{esc(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(rows)
    tail = (
        f'<p class="note">Showing {max_rows} of {len(df)} rows.</p>'
        if len(df) > max_rows
        else ""
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{tail}"


def cards(items: Sequence[tuple[str, str, str]]) -> str:
    """Key-figure cards. Each item is ``(label, value, css_class)``."""
    inner = "".join(
        f'<div class="card"><div class="k">{esc(k)}</div>'
        f'<div class="v {cls}">{esc(v)}</div></div>'
        for k, v, cls in items
    )
    return f'<div class="cards">{inner}</div>'


def figure_to_data_uri(fig) -> str:
    """Serialise a matplotlib figure to an inline PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def line_chart(
    x, series: dict[str, Sequence[float]], title: str = "", xlabel: str = "",
    ylabel: str = "", figsize: tuple[float, float] = (9.0, 3.6)
) -> str:
    """Render a line chart, or return an empty string if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plots are optional by design
        return ""

    fig, ax = plt.subplots(figsize=figsize)
    for label, values in series.items():
        ax.plot(x, values, linewidth=1.4, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    if len(series) > 1:
        ax.legend(fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    uri = figure_to_data_uri(fig)
    plt.close(fig)
    return f'<img alt="{esc(title)}" src="{uri}">'


def scatter_chart(
    x, y, title: str = "", xlabel: str = "", ylabel: str = "", diagonal: bool = False,
    figsize: tuple[float, float] = (5.4, 4.6)
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return ""

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=22, alpha=0.8)
    if diagonal:
        lo = min(min(x), min(y))
        hi = max(max(x), max(y))
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0, color="#888")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    uri = figure_to_data_uri(fig)
    plt.close(fig)
    return f'<img alt="{esc(title)}" src="{uri}">'


@dataclass
class Section:
    title: str
    blocks: list[str] = field(default_factory=list)

    def text(self, body: str) -> Section:
        self.blocks.append(f"<p>{esc(body)}</p>")
        return self

    def html(self, raw: str) -> Section:
        if raw:
            self.blocks.append(raw)
        return self

    def note(self, body: str) -> Section:
        self.blocks.append(f'<p class="note">{esc(body)}</p>')
        return self

    def pre(self, body: str) -> Section:
        self.blocks.append(f"<pre>{esc(body)}</pre>")
        return self

    def table(self, df: pd.DataFrame, max_rows: int = 60) -> Section:
        self.blocks.append(frame_to_html(df, max_rows))
        return self

    def cards(self, items: Sequence[tuple[str, str, str]]) -> Section:
        self.blocks.append(cards(items))
        return self

    def subheading(self, text: str) -> Section:
        self.blocks.append(f"<h3>{esc(text)}</h3>")
        return self

    def render(self) -> str:
        return f"<h2>{esc(self.title)}</h2>\n" + "\n".join(self.blocks)


@dataclass
class Report:
    title: str
    subtitle: str = ""
    sections: list[Section] = field(default_factory=list)
    footer: str = ""

    def section(self, title: str) -> Section:
        sec = Section(title)
        self.sections.append(sec)
        return sec

    def render(self) -> str:
        body = "\n".join(s.render() for s in self.sections)
        foot = f"<footer>{esc(self.footer)}</footer>" if self.footer else ""
        return (
            "<!doctype html>\n<html lang=\"en\">\n<head>\n"
            f'<meta charset="utf-8">\n<meta name="viewport" '
            f'content="width=device-width, initial-scale=1">\n'
            f"<title>{esc(self.title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
            f"<h1>{esc(self.title)}</h1>\n"
            + (f'<p class="sub">{esc(self.subtitle)}</p>\n' if self.subtitle else "")
            + body
            + "\n"
            + foot
            + "\n</body>\n</html>\n"
        )

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8")
        return target
