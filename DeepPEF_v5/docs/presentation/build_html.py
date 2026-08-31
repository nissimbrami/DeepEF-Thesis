"""Build a single self-contained HTML presentation from the 3 teaching markdown docs +
the generated figures. Figures are embedded as base64 so the .html is fully portable
(open in any browser, email it, no external files).

Run:  python DeepPEF_v5/docs/presentation/build_html.py
Out:  DeepPEF_v5/docs/presentation/DeepPEF_v5_Teaching.html
"""
import os
import base64

try:
    import markdown as md
except ImportError:
    raise SystemExit("pip install markdown  (needed to render the teaching docs)")

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")

DOC_FILES = [
    ("The Six Levers (A-F)", "TEACHING_LEVERS_AF.md"),
    ("Visual Representations", "TEACHING_VISUAL_REPRESENTATIONS.md"),
    ("Flow & Compatibility", "TEACHING_FLOW_AND_COMPATIBILITY.md"),
]

# Which figures head each section (by filename), with captions.
SECTION_FIGS = {
    "The Six Levers (A-F)": [
        ("fig1_dg_concept.png", "Figure 1 — folding free energy as an energy difference."),
        ("fig4_levers_grid.png", "Figure 4 — the six levers at a glance (default = OFF = baseline)."),
    ],
    "Visual Representations": [
        ("fig2_node_layout.png", "Figure 2 — node-feature layout and where each lever changes it."),
        ("fig3_architecture.png", "Figure 3 — the two-tower GCN+GATv2 pipeline."),
    ],
    "Flow & Compatibility": [
        ("fig5_flow_compose.png", "Figure 5 — the dimension contract and lever composability."),
    ],
}


def _img_tag(fname, caption):
    path = os.path.join(FIG, fname)
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


CSS = """
:root{--fg:#1a1a1a;--muted:#555;--acc:#DD8452;--blue:#4C72B0;--bg:#ffffff;--code:#f4f4f6;}
*{box-sizing:border-box;}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
 line-height:1.6;max-width:960px;margin:0 auto;padding:2rem 1.4rem 5rem;background:var(--bg);}
h1{font-size:2rem;border-bottom:3px solid var(--acc);padding-bottom:.3rem;margin-top:2.5rem;}
h2{font-size:1.5rem;color:var(--blue);margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.2rem;}
h3{font-size:1.2rem;margin-top:1.5rem;}
code{background:var(--code);padding:.15em .4em;border-radius:4px;font-size:.9em;
 font-family:SFMono-Regular,Consolas,monospace;}
pre{background:#2f2f2f;color:#f5f5f5;padding:1rem;border-radius:8px;overflow-x:auto;font-size:.85rem;}
pre code{background:none;color:inherit;padding:0;}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem;}
th,td{border:1px solid #ddd;padding:.5rem .7rem;text-align:left;vertical-align:top;}
th{background:#f0f2f6;}
blockquote{border-left:4px solid var(--acc);margin:1rem 0;padding:.5rem 1rem;background:#fff8f2;color:#333;}
figure{margin:1.5rem 0;text-align:center;}
figure img{max-width:100%;border:1px solid #e2e2e2;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.07);}
figcaption{color:var(--muted);font-size:.88rem;font-style:italic;margin-top:.5rem;}
.toc{background:#f7f8fa;border:1px solid #e4e6eb;border-radius:10px;padding:1rem 1.4rem;margin:1.5rem 0;}
.toc a{color:var(--blue);text-decoration:none;} .toc a:hover{text-decoration:underline;}
.cover{text-align:center;padding:3rem 1rem 1rem;}
.cover .sub{color:var(--muted);font-size:1.1rem;}
.badge{display:inline-block;background:var(--acc);color:#fff;border-radius:20px;
 padding:.2rem .8rem;font-size:.8rem;margin-top:.6rem;}
hr{border:none;border-top:1px solid #e4e6eb;margin:3rem 0;}
"""


def build():
    md_conv = md.Markdown(extensions=["tables", "fenced_code", "toc"])
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>DeepPEF_v5 — Teaching Presentation</title>",
             f"<style>{CSS}</style></head><body>"]

    parts.append(
        "<div class='cover'>"
        "<h1 style='border:none;font-size:2.4rem'>DeepPEF&nbsp;v5 — Teaching Presentation</h1>"
        "<div class='sub'>The six physics-grounded levers (A–F), the model representations, "
        "and how everything composes without breaking the dimension contract.</div>"
        "<div class='badge'>default = OFF = baseline (bit-for-bit)</div>"
        "</div>")

    # TOC
    parts.append("<div class='toc'><b>Contents</b><ol>")
    for title, _ in DOC_FILES:
        anchor = title.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("&", "")
        parts.append(f"<li><a href='#{anchor}'>{title}</a></li>")
    parts.append("</ol></div>")

    for title, fname in DOC_FILES:
        anchor = title.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("&", "")
        parts.append(f"<hr/><h1 id='{anchor}'>{title}</h1>")
        # section header figures
        for fig, cap in SECTION_FIGS.get(title, []):
            if os.path.exists(os.path.join(FIG, fig)):
                parts.append(_img_tag(fig, cap))
        with open(os.path.join(DOCS, fname), "r", encoding="utf-8") as fh:
            body = fh.read()
        md_conv.reset()
        parts.append(md_conv.convert(body))

    parts.append("</body></html>")
    out = os.path.join(HERE, "DeepPEF_v5_Teaching.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print("Wrote", out)


if __name__ == "__main__":
    build()
