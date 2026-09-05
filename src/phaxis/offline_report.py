"""Offline view over exported tables; never changes scientific results."""

from __future__ import annotations

import csv
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil

from .io import atomic_write_json

TABLE_NAMES = (
    "image_traits.csv",
    "traits.csv",
    "hair_instances.csv",
    "detailed_root_statistics.csv",
)


def _safe(value: str) -> str:
    if re.search(
        r"(?:[A-Za-z]:[\\/]|/(?:home|Users|data|mnt)/|(?:token|password|secret)\s*[=:])",
        value,
        re.I,
    ):
        return "[redacted local path or sensitive value]"
    return value


def build_report(traits: str | Path, output: str | Path, *, synthetic: bool = False) -> dict:
    traits, output = Path(traits), Path(output)
    if not (traits / "image_traits.csv").is_file():
        raise ValueError("image_traits.csv is required in --traits")
    if output.exists():
        raise FileExistsError(f"Choose a new report output directory: {output.name}")
    tables: dict = {}
    hashes: dict = {}
    # Read/validate all data before materializing a success-looking report.
    for name in TABLE_NAMES:
        path = traits / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Missing CSV header: {name}")
            rows = list(reader)
            if any(None in row or any(v is None for v in row.values()) for row in rows):
                raise ValueError(f"Malformed CSV row: {name}")
            # Avoid copying a sensitive string into downloadable tables as well.
            if any(_safe(v) != v for row in rows for v in row.values()):
                raise ValueError(
                    f"Sensitive local path or credential-like value in {name}; sanitize before reporting"
                )
            tables[name] = {"fields": reader.fieldnames, "rows": rows}
            # Show biological measurements before long identity/hash columns.
            front = [
                "task_id",
                "hair_count",
                "visible_hair_count",
                "hair_id",
                "length_um",
                "visible_root_axis_length_um",
                "median_root_width_um",
                "median_hair_length_um",
                "formal_statistics_eligible",
                "complete_length_measurement_eligible",
            ]
            fields = reader.fieldnames
            tables[name]["fields"] = [f for f in front if f in fields] + [
                f for f in fields if f not in front
            ]
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    (output / "tables").mkdir()
    for name in tables:
        shutil.copyfile(traits / name, output / "tables" / name)
    model = {
        "schema_version": "PHAxis-offline-report-1.0",
        "software_version": "1.0.0",
        "synthetic": synthetic,
        "tables": tables,
        "input_sha256": hashes,
    }
    atomic_write_json(output / "report_data.json", model)
    atomic_write_json(
        output / "provenance.json",
        {
            "software": "PHAxis",
            "version": "1.0.0",
            "input_sha256": hashes,
            "operation": "read_only_trait_report",
            "telemetry": False,
            "source_paths_included": False,
        },
    )
    roots = tables["image_traits.csv"]["rows"]
    hairs = tables.get("hair_instances.csv", {"rows": []})["rows"]
    eligible = sum(row.get("formal_statistics_eligible", "").lower() == "true" for row in roots)
    context = (
        "SYNTHETIC INSTALLATION EXAMPLE · not microscopy accuracy evidence"
        if synthetic
        else "EXPORTED TRAITS · inspect QC and observability before biological inference"
    )
    payload = (
        json.dumps(model, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    options = "".join(f'<option value="{escape(name)}">{escape(name)}</option>' for name in tables)
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PHAxis 1.0.0 | Trait report</title>
<style>
:root{--ink:#233949;--green:#176b63}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#f1f7f5;font:16px/1.5 system-ui,sans-serif}
main{max-width:1450px;margin:auto;padding:38px}header{border-bottom:3px solid var(--green);padding-bottom:20px;display:flex;justify-content:space-between;align-items:center;gap:20px}h1{font-size:42px;margin:0}h2{font-size:24px;margin:0 0 12px}.tag{color:var(--green);font-weight:700;letter-spacing:.06em;font-size:12px}.muted{color:#536774}section{margin:26px 0;background:white;border:1px solid #dbe6e2;border-radius:10px;padding:24px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}.card{padding:18px;background:white;border-radius:8px;border-top:4px solid var(--green)}.number{font-size:34px;font-weight:700;display:block}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:14px 0}input,select,button,a.download{font:inherit;padding:8px 12px;border:1px solid #b8cbc5;border-radius:5px;background:white;color:var(--ink)}a{color:var(--green)}.scroll{overflow:auto;max-height:440px;border:1px solid #dbe6e2}table{border-collapse:collapse;font-size:13px;min-width:100%}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e6edeb;white-space:nowrap}th{background:#e7f3ef;position:sticky;top:0;cursor:pointer}th:hover{background:#d4e9e1}.flow{display:flex;gap:12px;flex-wrap:wrap}.flow span{padding:12px 18px;background:#e7f3ef;border-radius:6px}.warn{color:#915300}.hash{font:12px monospace;overflow-wrap:anywhere}.foot{font-size:13px}.empty{padding:20px}
@media(max-width:750px){main{padding:16px}.cards{grid-template-columns:repeat(2,1fr)}header{display:block}h1{font-size:32px}section{padding:16px}}
@media print{body{background:white}main{padding:0}.controls{display:none}.scroll{max-height:none;overflow:visible}section{break-inside:avoid}.cards{grid-template-columns:repeat(4,1fr)}}
</style></head><body><main><header><div><div class="tag">ORGAN-REGISTERED PHENOTYPING</div><h1>PHAxis <small style="font-size:20px">1.0.0</small></h1><div class="muted">Primary-root geometry · hair identities · endpoint-supported morphology</div></div><div class="tag">OFFLINE REPORT<br>@@CONTEXT@@</div></header>
<div class="cards"><div class="card"><span class="number">@@ROOTS@@</span>Source roots</div><div class="card"><span class="number">@@HAIRS@@</span>Visible hair identities</div><div class="card"><span class="number">32</span>Canonical descriptors</div><div class="card"><span class="number">@@ELIGIBLE@@ / @@ROOTS@@</span>Formal-eligibility flags</div></div>
<section><h2>One root, multiple measurement layers</h2><div class="flow"><span>Root geometry</span><span>Hair identity</span><span>Complete-length linkage</span><span>Calibrated traits</span></div><p class="muted">The source root is the sampling unit. An observed hair can support a count without an endpoint-complete length. Missing length is not zero. Synthetic eligibility flags test code paths; they do not make a fixture biological evidence.</p></section>
<section><h2>Trait tables and measurement support</h2><div class="controls"><select id="tableChoice" aria-label="Select table">@@OPTIONS@@</select><input id="search" aria-label="Search rows" placeholder="Search any field"><a id="download" class="download" download>Download source CSV</a><button id="prev">Previous</button><button id="next">Next</button><span id="status"></span></div><p class="muted">Click a column heading to sort. Hover over a value for its exact contents. QC, reason and support fields remain available alongside the measurements.</p><div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div></section>
<section><h2>Provenance and interpretation</h2><p>Read-only report of exported PHAxis tables. No external scripts, fonts, CDNs, model downloads or telemetry. Downloads preserve source CSV bytes.</p><details><summary>Input SHA-256 and report data</summary><div id="hashes" class="hash"></div><p><a href="report_data.json" download>Report data JSON</a> · <a href="provenance.json" download>Provenance JSON</a></p></details><p class="foot">Software: PHAxis 1.0.0. Citation metadata and public DOI remain owner-confirmation items in this local release candidate. Source license: Apache-2.0.</p></section></main>
<script type="application/json" id="data">@@DATA@@</script><script>
const model=JSON.parse(document.getElementById('data').textContent);let page=0,sort='',ascending=true;const el=id=>document.getElementById(id);const size=50;
function render(){const name=el('tableChoice').value,t=model.tables[name],q=el('search').value.toLowerCase();let rows=t.rows.filter(r=>Object.values(r).some(v=>v.toLowerCase().includes(q)));if(sort)rows.sort((a,b)=>{let x=a[sort],y=b[sort];const num=x!==''&&y!==''&&Number.isFinite(Number(x))&&Number.isFinite(Number(y));return (num?Number(x)-Number(y):x.localeCompare(y))*(ascending?1:-1)});const pages=Math.max(1,Math.ceil(rows.length/size));page=Math.min(page,pages-1);el('head').replaceChildren();const tr=document.createElement('tr');for(const f of t.fields){const th=document.createElement('th');th.textContent=f+(sort===f?(ascending?' ↑':' ↓'):'');th.onclick=()=>{ascending=sort===f?!ascending:true;sort=f;render()};tr.append(th)}el('head').append(tr);el('body').replaceChildren();for(const r of rows.slice(page*size,(page+1)*size)){const tr=document.createElement('tr');for(const f of t.fields){const td=document.createElement('td');td.textContent=r[f]||'—';td.title=r[f]||'Missing / not supported';tr.append(td)}el('body').append(tr)}el('status').textContent=rows.length+' rows · page '+(page+1)+' / '+pages;el('prev').disabled=page===0;el('next').disabled=page===pages-1;el('download').href='tables/'+name}
el('tableChoice').onchange=()=>{page=0;sort='';render()};el('search').oninput=()=>{page=0;render()};el('prev').onclick=()=>{page--;render()};el('next').onclick=()=>{page++;render()};for(const [n,h] of Object.entries(model.input_sha256)){const p=document.createElement('p');p.textContent=n+': '+h;el('hashes').append(p)}render();
</script></body></html>"""
    for key, value in {
        "CONTEXT": escape(context),
        "ROOTS": str(len(roots)),
        "HAIRS": str(len(hairs)),
        "ELIGIBLE": str(eligible),
        "OPTIONS": options,
        "DATA": payload,
    }.items():
        html = html.replace(f"@@{key}@@", value)
    (output / "report.html").write_text(html, encoding="utf-8")
    (output / "README_results.md").write_text(
        "# PHAxis report results\n\nOpen report.html offline. tables/ holds byte-identical source CSVs; "
        "report_data.json is the versioned table model; provenance.json records SHA-256. "
        "Units and missing-value meanings are defined in the PHAxis trait dictionary. "
        "QC/eligibility and observability are retained, not reclassified.\n",
        encoding="utf-8",
    )
    return {
        "status": "passed",
        "tables": len(tables),
        "source_roots": len(roots),
        "synthetic": synthetic,
    }
