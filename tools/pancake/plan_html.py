#!/usr/bin/env python3
"""plan_html.py - render the development plan (MOD_PLAN.md phases 16 and 15, TEST_PLAN.md tiers and phase 4) as one
self-contained HTML page to read locally: what is proposed, why, and where the gain is.

    python3 tools/pancake/plan_html.py [--out ~/pancake-plan/plan.html]

The page is generated from the markdown, so re-run it after the plan changes.
"""
import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def section(text, start, stop):
    a = text.index(start)
    b = text.find(stop, a + len(start)) if stop else -1
    return text[a:b if b > 0 else len(text)]


def inline(s):
    """Minimal markdown inline -> HTML: code, bold, links, bare URLs."""
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2" target="_blank">\1</a>', s)
    s = re.sub(r'(?<!href=")(?<![">])(https?://[^\s<)|,;]+)', r'<a href="\1" target="_blank">link</a>', s)
    return s


def parse_blocks(md):
    """Split a markdown section into subsections: [(title, [paragraph lines], [tables], [list items])]."""
    subs, cur = [], None
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### "):
            cur = {"title": ln[4:].strip(), "text": [], "tables": [], "items": []}
            subs.append(cur)
        elif cur is None:
            pass
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            cur["tables"].append(rows)
            continue
        elif re.match(r"^\s*(\d+\.|-)\s", ln):
            item = re.sub(r"^\s*(\d+\.|-)\s", "", ln)
            while i + 1 < len(lines) and lines[i + 1].startswith("   ") and not re.match(r"^\s*(\d+\.|-)\s", lines[i + 1]):
                i += 1
                item += " " + lines[i].strip()
            cur["items"].append(item)
        elif ln.strip():
            cur["text"].append(ln.strip())
        i += 1
    return subs


def intro(md):
    out = []
    for ln in md.splitlines()[1:]:
        if ln.startswith("###"):
            break
        out.append(ln.strip())
    return " ".join(x for x in out if x)


def cls_for(col, val):
    v = val.lower()
    if col == "exact":
        return "lossy" if "lossy" in v else "rounding" if "round" in v else "exact" if "exact" in v else ""
    if col == "status":
        return "done" if "done" in v else "rejected" if "reject" in v else "new" if v.startswith("new") else "planned"
    return ""


def table_html(rows, area):
    head = rows[0]
    keys = [h.lower() for h in head]
    idx = {k: next((i for i, h in enumerate(keys) if k in h), None)
           for k in ("exact", "gpu", "tier", "status", "gain", "effort")}
    out = ['<div class="tw"><table><thead><tr>' + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows[1:]:
        r = r + [""] * (len(head) - len(r))
        attrs = {k: (r[i].lower() if i is not None else "") for k, i in idx.items()}
        data = " ".join(f'data-{k}="{html.escape(v)}"' for k, v in attrs.items())
        tds = []
        for i, c in enumerate(r[:len(head)]):
            k = next((k for k, j in idx.items() if j == i), "")
            extra = cls_for(k, c)
            klass = f' class="{k} {extra}"' if k else ""
            tds.append(f"<td{klass}>{inline(c)}</td>")
        out.append(f'<tr data-area="{html.escape(area)}" {data}>' + "".join(tds) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def render_phase(md, pid):
    subs = parse_blocks(md)
    parts = [f'<p class="lead">{inline(intro(md))}</p>']
    nav = []
    for n, s in enumerate(subs):
        sid = f"{pid}-{n}"
        count = sum(len(t) - 1 for t in s["tables"])
        nav.append(f'<a href="#{sid}">{inline(s["title"])}{f" <span class=cnt>{count}</span>" if count else ""}</a>')
        body = "".join(f"<p>{inline(' '.join(s['text']))}</p>" for _ in [0] if s["text"])
        body += "".join(table_html(t, s["title"]) for t in s["tables"])
        if s["items"]:
            tag = "ol" if "top" in s["title"].lower() else "ul"
            body += f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in s["items"]) + f"</{tag}>"
        parts.append(f'<section class="sub" id="{sid}"><h3>{inline(s["title"])}</h3>{body}</section>')
    return nav, "\n".join(parts)


CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1f;--mut:#5f6368;--card:#fff;--line:#e3e3e0;--acc:#2b59c3;--ok:#1f7a3f;--warn:#9a5b00;--bad:#b3261e;--hi:#fff6d6}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#e8e8ea;--mut:#a0a3a8;--card:#1e1f22;--line:#33353a;--acc:#8ab4f8;--ok:#7fd49b;--warn:#f0b960;--bad:#f28b82;--hi:#3a3420}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:28px 32px 12px;max-width:1500px;margin:auto}h1{margin:0 0 6px;font-size:26px}header p{color:var(--mut);margin:4px 0}
.layout{display:grid;grid-template-columns:290px 1fr;gap:24px;max-width:1500px;margin:auto;padding:0 32px 60px}
nav{position:sticky;top:12px;align-self:start;max-height:calc(100vh - 24px);overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
nav h4{margin:10px 0 4px;font-size:12px;text-transform:uppercase;color:var(--mut);letter-spacing:.04em}nav a{display:block;padding:3px 6px;border-radius:6px;color:var(--fg);text-decoration:none;font-size:13.5px}
nav a:hover{background:var(--hi)}.cnt{float:right;color:var(--mut);font-size:12px}
main section.phase{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin-bottom:24px}
h2{margin:0 0 8px;font-size:21px}h3{margin:22px 0 8px;font-size:17px;border-top:1px solid var(--line);padding-top:16px}
.lead{color:var(--mut)}code{background:rgba(127,127,127,.15);padding:1px 4px;border-radius:4px;font-size:12.5px}
a{color:var(--acc)}.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--card);font-weight:600}
td.gain{background:var(--hi);font-weight:500;min-width:180px}td.exact.exact{color:var(--ok)}td.exact.lossy{color:var(--bad)}td.exact.rounding{color:var(--warn)}
td.status.done{color:var(--ok);font-weight:600}td.status.rejected{color:var(--mut);text-decoration:line-through}td.status.new{color:var(--acc);font-weight:600}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 18px;padding:12px;background:var(--card);border:1px solid var(--line);border-radius:10px;position:sticky;top:0;z-index:5}
.filters input,.filters select{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font-size:13px}
.filters input{flex:1;min-width:200px}.pill{font-size:12px;color:var(--mut)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin:14px 0}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--bg)}.card b{display:block;margin-bottom:4px}.card span{color:var(--mut);font-size:13px}
tr.hide{display:none}ol li,ul li{margin:6px 0}
@media (max-width:900px){.layout{grid-template-columns:1fr;padding:0 16px 40px}nav{position:static;max-height:none}header{padding:20px 16px 8px}}
"""

JS = """
const q=document.getElementById('q'),fe=document.getElementById('fexact'),fg=document.getElementById('fgpu'),fs=document.getElementById('fstatus'),ft=document.getElementById('ftier'),cnt=document.getElementById('shown');
function apply(){const t=q.value.toLowerCase();let n=0;document.querySelectorAll('tbody tr').forEach(r=>{
 const ok=(!t||r.textContent.toLowerCase().includes(t))&&(!fe.value||(r.dataset.exact||'').includes(fe.value))&&(!fg.value||(r.dataset.gpu||'').includes(fg.value))
 &&(!fs.value||(r.dataset.status||'').includes(fs.value))&&(!ft.value||(r.dataset.tier||'').includes(ft.value));r.classList.toggle('hide',!ok);if(ok)n++});cnt.textContent=n+' rows shown';}
[q,fe,fg,fs,ft].forEach(e=>e.addEventListener('input',apply));apply();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "pancake-plan" / "plan.html"))
    a = ap.parse_args()
    mod = (ROOT / "MOD_PLAN.md").read_text()
    tp = (ROOT / "bench/TEST_PLAN.md").read_text()
    p16 = section(mod, "## Phase 16:", "\n## Phase 15:")
    p15 = section(mod, "## Phase 15:", "\n## Phase 12")
    tiers = section(tp, "## How every test runs", "\n## The optimization loop")
    p4 = section(tp, "## Phase 4 -", None)

    nav16, body16 = render_phase(p16, "p16")
    nav15, body15 = render_phase(p15, "p15")
    # the tier section has a table and rules but no ### headings: wrap it
    tiers_md = "x\n### Test tiers\n" + "\n".join(tiers.splitlines()[1:])
    navt, bodyt = render_phase(tiers_md, "tiers")
    nav4, body4 = render_phase(p4, "p4")

    top = parse_blocks(p16)
    top10 = next((s["items"] for s in top if "top 10" in s["title"].lower()), [])
    areas = [s for s in top if s["tables"] and "top" not in s["title"].lower()]
    cards = "".join(
        f'<a class="card" href="#p16-{top.index(s)}" style="text-decoration:none;color:inherit"><b>{inline(s["title"])}</b>'
        f'<span>{sum(len(t) - 1 for t in s["tables"])} items</span></a>' for s in areas)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pancake development plan</title><style>{CSS}</style></head><body>
<header><h1>Pancake-strata development plan</h1>
<p>What is proposed, why, and where the gain is. Generated from <code>MOD_PLAN.md</code> (phases 16 and 15) and
<code>bench/TEST_PLAN.md</code> (test tiers, phase 4). Regenerate with <code>python3 tools/pancake/plan_html.py</code>.</p>
<p>How to read it: <b>Expected gain</b> (yellow column) is the paper's or fork author's claim on their hardware, or our
estimate, unless it says <i>meas.</i>; nothing else was measured on our box. <span style="color:var(--ok)">exact</span> = same
text; <span style="color:var(--warn)">rounding</span> = exact math, last bits may move; <span style="color:var(--bad)">lossy</span> =
text may change (needs the quality gate). Order: academic technologies first (phase 16), then fork ideas (phase 15).</p></header>
<div class="layout"><nav>
<h4>Start here</h4><a href="#overview">Overview: where the gain is</a><a href="#top10">Top 10 to develop next</a>
<h4>Phase 16 - academic first</h4>{''.join(nav16)}
<h4>Phase 15 - fork ideas</h4>{''.join(nav15)}
<h4>Testing</h4>{''.join(navt)}{''.join(nav4)}
</nav><main>
<div class="filters"><input id="q" placeholder="Search technology, paper, switch, fork..."><select id="fexact"><option value="">exact / lossy</option><option value="exact">exact</option><option value="round">rounding</option><option value="lossy">lossy</option></select>
<select id="fgpu"><option value="">any GPU</option><option value="both">both</option><option value="hip">HIP / 7900 XT</option><option value="cuda">CUDA / 3060</option></select>
<select id="fstatus"><option value="">any status</option><option value="new">new</option><option value="planned">planned</option><option value="done">done</option><option value="reject">rejected</option></select>
<select id="ftier"><option value="">any test tier</option><option value="offline">offline</option><option value="screen">screen</option><option value="confirm">confirm</option><option value="rigorous">rigorous</option></select>
<span class="pill" id="shown"></span></div>
<section class="phase" id="overview"><h2>Overview: where the gain is</h2>
<p>Decode is bound by reading expert bytes (CPU experts from DDR4, GPU experts from VRAM), and agent turns lose time to
prompt re-reads. So the large gains are in: fewer expert bytes on the critical path (routing, placement, overlap), no
repeated prompt reads between turns, and more accepted tokens per verify window. GPU kernel work is a smaller,
mostly exact floor (+2-5%). The areas below are ordered by total expected gain.</p>
<div class="cards">{cards}</div></section>
<section class="phase" id="top10"><h2>Top 10 academic items to develop next</h2><ol>{''.join(f"<li>{inline(x)}</li>" for x in top10)}</ol></section>
<section class="phase" id="phase16"><h2>Phase 16 - academic research first, by area of improvement</h2>{body16}</section>
<section class="phase" id="phase15"><h2>Phase 15 - speed ideas from all fork commits</h2>{body15}</section>
<section class="phase" id="testing"><h2>Testing: tiers (rigorous only where needed)</h2>{bodyt}</section>
<section class="phase" id="phase4"><h2>Test order for the new ideas (TEST_PLAN phase 4)</h2>{body4}</section>
</main></div><script>{JS}</script></body></html>"""
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main()
