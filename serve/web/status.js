// The status page's live part (serve/web/status.html): every 2 s it asks GET /status.json and redraws the sections from
// its "view" (the cells are formatted on the server, serve/status_page.py, so the page reads the same with JavaScript off).
// Read-only: GET only. The cells are set as text, never as markup.
"use strict";
const $ = id => document.getElementById(id);
let apiKey = "";
try { apiKey = sessionStorage.getItem("strata.monitor.key") || ""; } catch (e) { /* ignore */ }
let paused = false, timer = null;

function el(tag, attrs, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function section(s) {
  const box = el("section", {class: "panel", id: "s-" + s.id});
  const bar = el("div", {class: "panel-bar"});
  bar.append(el("h2", {}, s.title), el("span", {class: "small"}, s.note || ""));
  const table = el("table", {class: s.kind});
  const headRow = el("tr");
  for (const c of s.columns) headRow.append(el("th", {scope: "col"}, c));
  const thead = el("thead"); thead.append(headRow);
  const tbody = el("tbody");
  if (s.kind === "kv") {
    for (const r of s.rows) { const tr = el("tr"); tr.append(el("th", {scope: "row"}, r[0]), el("td", {}, r[1])); tbody.append(tr); }
  } else if (s.rows.length) {
    for (const r of s.rows) { const tr = el("tr"); for (const c of r) tr.append(el("td", {}, c)); tbody.append(tr); }
  } else {
    const tr = el("tr"); tr.append(el("td", {colspan: String(s.columns.length), class: "empty"}, s.empty || "—")); tbody.append(tr);
  }
  table.append(thead, tbody);
  const scroll = el("div", {class: "scroll"}); scroll.append(table);
  box.append(bar, scroll);
  return box;
}

function render(view) {
  const box = $("sections");
  box.replaceChildren(...view.map(section));
}

async function refresh() {
  try {
    const r = await fetch("/status.json", {cache: "no-store", headers: apiKey ? {Authorization: "Bearer " + apiKey} : {}});
    if (r.status === 401) { $("auth").hidden = false; $("error").hidden = true; return; }
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    render(d.view || []);
    $("error").hidden = true;
    $("auth").hidden = true;
    $("updated").textContent = "Updated " + new Date().toLocaleTimeString() + (paused ? " · paused" : " · every 2 s");
  } catch (e) {
    $("error").textContent = "The server did not answer: " + e.message;
    $("error").hidden = false;
    $("updated").textContent = "Last update failed";
  }
}

function schedule() {
  clearTimeout(timer);
  if (!paused) timer = setTimeout(async () => { await refresh(); schedule(); }, 2000);
}

$("controls").hidden = false;
$("pause").onclick = () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  $("pause").setAttribute("aria-label", paused ? "Resume live updates" : "Pause live updates");
  if (paused) clearTimeout(timer); else schedule();
};
$("refresh").onclick = async () => { await refresh(); schedule(); };
$("key").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("theme").onclick = () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("strata.theme", theme); } catch (e) { /* ignore */ }
};
$("auth").onsubmit = async event => {
  event.preventDefault();
  apiKey = $("api-key").value.trim();
  try { sessionStorage.setItem("strata.monitor.key", apiKey); } catch (e) { /* ignore */ }
  $("auth").hidden = true;
  await refresh();
  schedule();
};

refresh();
schedule();
