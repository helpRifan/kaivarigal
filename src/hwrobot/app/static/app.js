// Kaivarigal web UI. Plain ES module, no build step, no network beyond this server.

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const view = $("#view");

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const ICON = {
  play: '<svg viewBox="0 0 24 24"><path d="M7 5l12 7-12 7z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>',
  stop: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>',
  trace: '<svg viewBox="0 0 24 24"><rect x="4" y="6" width="16" height="12" rx="1" stroke-dasharray="3 2.5"/></svg>',
  up: '<svg viewBox="0 0 24 24"><path d="M6 15l6-6 6 6"/></svg>',
  down: '<svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>',
  file: '<svg viewBox="0 0 24 24"><path d="M14 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V7z"/><path d="M14 3v4h4M9 13h6M9 17h6"/></svg>',
  camera: '<svg viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>',
  upload: '<svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg>',
  shuffle: '<svg viewBox="0 0 24 24"><path d="M4 7h3c5 0 5 10 10 10h3M4 17h3c2 0 3-1.5 4-3.5M17 7h3M17 4l3 3-3 3M17 14l3 3-3 3"/></svg>',
  undo: '<svg viewBox="0 0 24 24"><path d="M9 7L4 12l5 5"/><path d="M4 12h10a6 6 0 0 1 0 12"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>',
};

// ------------------------------------------------------------------ api + toast

async function api(path, { method, body, form } = {}) {
  const opt = { method: method || (body !== undefined || form ? "POST" : "GET") };
  if (body !== undefined) {
    opt.headers = { "Content-Type": "application/json" };
    opt.body = JSON.stringify(body);
  }
  if (form) opt.body = form;
  const res = await fetch(path, opt);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const d = data?.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data;
}

let toastTimer;
function toast(msg, kind = "") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.className = "toast"), kind === "error" ? 5000 : 2600);
}
const fail = (e) => toast(e.message || String(e), "error");

// ------------------------------------------------------------------ store

const saved = (() => { try { return JSON.parse(localStorage.getItem("kaivarigal") || "{}"); } catch { return {}; } })();

const store = {
  page: null,
  pageId: saved.pageId || null,
  assignments: saved.assignments || {},
  selected: saved.selected || null,
  style: saved.style || null,
  profile: saved.profile || "normal",
  skipUsed: saved.skipUsed ?? true,
  feedRate: saved.feedRate || 1500,
  tab: saved.tab || "text",
  zoom: false,
  presets: [],
  preview: null,
  job: null,
  status: null,
};

function persist() {
  try {
    const { pageId, assignments, selected, style, profile, skipUsed, feedRate, tab } = store;
    localStorage.setItem("kaivarigal", JSON.stringify({ pageId, assignments, selected, style, profile, skipUsed, feedRate, tab }));
  } catch { /* storage blocked: fine */ }
}

async function loadPresets() {
  if (store.presets.length) return;
  store.presets = await api("/api/styles");
  if (!store.style) store.style = { ...store.presets[0].style };
}

async function ensurePage() {
  if (store.page) return store.page;
  if (store.pageId) {
    try { return setPage(await api(`/api/pages/${store.pageId}`), false); } catch { /* server restarted */ }
  }
  return setPage(await api("/api/pages/from-template", { body: { template_id: "ruled-8" } }), true);
}

function setPage(page, reset = true) {
  store.page = page;
  store.pageId = page.id;
  if (reset) {
    store.assignments = {};
    store.selected = page.lines.find((l) => !l.has_writing)?.id || page.lines[0]?.id || null;
  }
  const ids = new Set(page.lines.map((l) => l.id));
  for (const k of Object.keys(store.assignments)) if (!ids.has(k)) delete store.assignments[k];
  if (store.selected && !ids.has(store.selected)) store.selected = page.lines[0]?.id || null;
  store.preview = null;
  persist();
  return page;
}

// ------------------------------------------------------------------ status chip + job polling

const statusEl = $("#status");
function renderStatus() {
  const s = store.status;
  const job = store.job && ["running", "paused"].includes(store.job.state) ? store.job : s?.job;
  let state = "idle", text = "Simulator · Ready";
  if (!s) { state = "offline"; text = "Server offline"; }
  else if (job?.state === "running") { state = "running"; text = `${job.trace_only ? "Tracing" : "Writing"} · ${Math.round(job.progress * 100)}%`; }
  else if (job?.state === "paused") { state = "paused"; text = `Paused · ${Math.round(job.progress * 100)}%`; }
  statusEl.dataset.state = state;
  $(".status-text", statusEl).textContent = text;
}

async function pollStatus() {
  try { store.status = await api("/api/status"); }
  catch { store.status = null; }
  if (!store.job && store.status?.job) { store.job = store.status.job; watchJob(); }
  renderStatus();
}
setInterval(pollStatus, 4000);

let jobTimer = null;
function watchJob() {
  if (jobTimer) return;
  jobTimer = setInterval(async () => {
    if (!store.job) return;
    try { store.job = await api(`/api/jobs/${store.job.id}`); }
    catch (e) { clearInterval(jobTimer); jobTimer = null; return fail(e); }
    renderStatus();
    onJobUpdate();
    if (["done", "stopped"].includes(store.job.state)) {
      clearInterval(jobTimer); jobTimer = null;
      const j = store.job;
      if (j.trace_only) toast(j.state === "done" ? "Trace finished. Check the pen covers the right area." : "Trace stopped.");
      else toast(j.state === "done" ? "Page written." : "Stopped. The partial page is in History.");
      setTimeout(() => { if (store.job === j) { store.job = null; onJobUpdate(); renderStatus(); } }, j.trace_only ? 600 : 2500);
    }
  }, 200);
}

let onJobUpdate = () => {};

// ------------------------------------------------------------------ router

const routes = { write: viewWrite, scan: viewScan, capture: viewCapture, history: viewHistory, calibrate: viewCalibrate };
let cleanup = () => {};

function route() {
  const name = (location.hash.replace(/^#\/?/, "") || "write").split("?")[0];
  const fn = routes[name] || viewWrite;
  cleanup(); cleanup = () => {}; onJobUpdate = () => {};
  $$(".nav a").forEach((a) => {
    if (a.dataset.route === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  view.innerHTML = "";
  Promise.resolve(fn()).catch(fail);
}
window.addEventListener("hashchange", route);

// ================================================================== WRITE

const lineNum = (id) => id.replace(/^L/, "");

async function viewWrite() {
  view.innerHTML = `
    <section class="write">
      <div class="stage">
        <div class="stage-bar" id="stagebar"></div>
        <div class="page-wrap" id="pagewrap"><div class="muted">Loading page…</div></div>
      </div>
      <aside class="panel" aria-label="Writing controls">
        <div class="seg" role="tablist" id="tabs">
          ${["text", "style", "plot"].map((t) => `<button role="tab" data-tab="${t}" aria-selected="${store.tab === t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join("")}
        </div>
        <div class="panel-body" id="tabbody" role="tabpanel"></div>
      </aside>
    </section>`;
  $("#tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (!b) return;
    store.tab = b.dataset.tab; persist();
    $$("#tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
    renderTab();
  });
  await Promise.all([loadPresets(), ensurePage()]);
  drawStageBar();
  drawPage();
  renderTab();
  onJobUpdate = () => { drawInkProgress(); if (store.tab === "plot") renderPlot(); };
  await refreshPreview();
}

function drawStageBar() {
  const p = store.page, s = store.status;
  const used = p.lines.filter((l) => l.has_writing).length;
  $("#stagebar").innerHTML = `
    <h1>${esc(p.label)}</h1>
    <span class="badge num">${p.lines.length} lines${used ? ` · ${used} already written` : ""}</span>
    ${p.simulated ? '<span class="badge fake" title="Lines were guessed, not detected">Lines guessed</span>' : ""}
    ${s?.model_simulated !== false ? '<span class="badge fake" title="The handwriting model is not trained yet">Placeholder handwriting</span>' : ""}
    <span style="margin-left:auto"></span>
    <button class="btn ghost small" id="zoom" aria-pressed="${!!store.zoom}">${store.zoom ? "Fit page" : "Zoom in"}</button>
    <a class="btn ghost small" href="#/scan">Change page</a>`;
  $("#zoom").onclick = () => {
    store.zoom = !store.zoom;
    applyZoom();
    drawStageBar();
  };
  applyZoom();
}

function applyZoom() {
  const wrap = $("#pagewrap");
  if (!wrap) return;
  wrap.classList.toggle("zoomed", !!store.zoom);
  if (store.zoom) scrollToSelected();
}

function scrollToSelected() {
  const g = store.selected && $(`#rules .rule[data-id="${store.selected}"]`);
  if (store.zoom && g) g.querySelector(".rule-hit").scrollIntoView({ block: "center", inline: "nearest" });
}

function drawPage() {
  const p = store.page;
  const sp = p.line_spacing_mm || 8;
  const rules = p.lines.map((l) => {
    const cls = ["rule", store.assignments[l.id]?.trim() ? "assigned" : "", store.selected === l.id ? "selected" : ""].join(" ");
    return `<g class="${cls}" data-id="${l.id}">
      <rect class="rule-band" x="0" y="${l.y_mm - sp + 0.5}" width="${p.width_mm}" height="${sp - 0.5}"/>
      <rect class="rule-mark" x="${l.x_start_mm - 1.6}" y="${l.y_mm - sp * 0.62}" width="0.9" height="${sp * 0.5}" rx="0.45"/>
      <text class="rule-num" x="${Math.max(2, (p.margin_x_mm || l.x_start_mm) - 7)}" y="${l.y_mm - 1}">${lineNum(l.id)}${l.has_writing ? "·" : ""}</text>
      <rect class="rule-hit" x="0" y="${l.y_mm - sp + 0.5}" width="${p.width_mm}" height="${sp}"><title>Line ${lineNum(l.id)}</title></rect>
    </g>`;
  }).join("");
  $("#pagewrap").innerHTML = `
    <svg id="pagesvg" viewBox="0 0 ${p.width_mm} ${p.height_mm}" role="img" aria-label="Page preview. Tap a line to write on it.">
      <image href="${p.image_url}" width="${p.width_mm}" height="${p.height_mm}" preserveAspectRatio="none"/>
      <g id="rules">${rules}</g>
      <g class="ink" id="ink"></g>
      <g id="trace"></g>
    </svg>`;
  $("#rules").addEventListener("click", (e) => {
    const g = e.target.closest(".rule");
    if (g) selectLine(g.dataset.id, true);
  });
  drawInk();
}

function refreshRuleClasses() {
  $$("#rules .rule").forEach((g) => {
    g.classList.toggle("selected", g.dataset.id === store.selected);
    g.classList.toggle("assigned", !!store.assignments[g.dataset.id]?.trim());
  });
}

function selectLine(id, focus = false) {
  store.selected = id; persist();
  refreshRuleClasses();
  scrollToSelected();
  if (store.tab !== "text") {
    store.tab = "text";
    $$("#tabs button").forEach((x) => x.setAttribute("aria-selected", x.dataset.tab === "text"));
  }
  renderTab();
  if (focus && matchMedia("(pointer: fine)").matches) $("#linetext")?.focus();
}

function drawInk() {
  const ink = $("#ink");
  if (!ink) return;
  const pv = store.preview;
  ink.innerHTML = pv ? pv.lines.flatMap((l) => l.paths.map((d) => `<path d="${d}"/>`)).join("") : "";
  inkLengths = null;
  drawInkProgress();
}

let inkLengths = null;
function drawInkProgress() {
  const ink = $("#ink"), trace = $("#trace");
  if (!ink) return;
  const job = store.job;
  const active = job && ["running", "paused"].includes(job.state);
  trace.innerHTML = "";
  if (active && job.trace_only && store.preview?.bbox) {
    const [x0, y0, x1, y1] = store.preview.bbox;
    trace.innerHTML = `<rect class="trace-rect" x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}"/>`;
  }
  const plotting = job && !job.trace_only && job.page_id === store.page?.id && (active || job.state !== "done");
  ink.classList.toggle("plotting", !!plotting);
  const paths = $$("path", ink);
  if (!plotting) { paths.forEach((p) => { p.classList.remove("drawn"); p.style.strokeDasharray = ""; }); return; }
  if (!inkLengths) inkLengths = paths.map((p) => p.getTotalLength());
  let left = job.drawn_mm;
  paths.forEach((p, i) => {
    const L = inkLengths[i];
    if (left >= L) { p.classList.add("drawn"); p.style.strokeDasharray = ""; }
    else if (left > 0) { p.classList.add("drawn"); p.style.strokeDasharray = `${left} ${L + 1}`; }
    else { p.classList.remove("drawn"); p.style.strokeDasharray = ""; }
    left -= L;
  });
}

let previewTimer, previewSeq = 0;
function schedulePreview(ms = 250) {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreview, ms);
}

function writePayload() {
  const assignments = {};
  for (const [k, v] of Object.entries(store.assignments)) if (v.trim()) assignments[k] = v;
  return { page_id: store.page.id, assignments, style: store.style, skip_used: store.skipUsed };
}

async function refreshPreview() {
  if (!store.page) return;
  const seq = ++previewSeq;
  try {
    const pv = await api("/api/preview", { body: writePayload() });
    if (seq !== previewSeq) return;
    store.preview = pv;
    drawInk();
    renderWarnings();
    if (store.tab === "plot") renderPlot();
  } catch (e) {
    if (/Page not found/.test(e.message)) { store.page = null; store.pageId = null; await ensurePage(); drawStageBar(); drawPage(); return refreshPreview(); }
    fail(e);
  }
}

function renderTab() {
  if (!$("#tabbody")) return;
  ({ text: renderText, style: renderStyle, plot: renderPlot })[store.tab]();
}

function renderWarnings() {
  const box = $("#warnings");
  if (!box) return;
  const w = store.preview?.warnings || [];
  box.innerHTML = w.length ? `<div class="notice" role="alert"><ul>${w.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>` : "";
  renderAssigned();
}

// ---- Text tab
function renderText() {
  const p = store.page;
  const line = p.lines.find((l) => l.id === store.selected);
  const idx = line ? p.lines.indexOf(line) : -1;
  $("#tabbody").innerHTML = line ? `
    <div class="line-head">
      <div>
        <h2>Line ${lineNum(line.id)} <span class="muted small num">of ${p.lines.length}</span></h2>
        ${line.has_writing ? '<p class="small" style="color:var(--warn)">This line already has writing on the page.</p>' : ""}
      </div>
      <div class="line-nav">
        <button class="btn" id="prevline" aria-label="Previous line" ${idx <= 0 ? "disabled" : ""}>${ICON.up}</button>
        <button class="btn" id="nextline" aria-label="Next line" ${idx >= p.lines.length - 1 ? "disabled" : ""}>${ICON.down}</button>
      </div>
    </div>
    <textarea id="linetext" aria-label="Text for line ${lineNum(line.id)}" placeholder="Type what goes on this line. Longer text carries on to the next empty lines.">${esc(store.assignments[line.id] || "")}</textarea>
    <div class="row">
      <label class="btn file-btn">${ICON.file} Import .txt<input type="file" id="importtxt" accept=".txt,text/plain"></label>
      <button class="btn ghost" id="clearline" ${store.assignments[line.id] ? "" : "disabled"}>Clear line</button>
    </div>
    <label class="check"><input type="checkbox" id="skipused" ${store.skipUsed ? "checked" : ""}> Skip lines that already have writing</label>
    <div id="warnings"></div>
    <div class="stack" id="assignedwrap"></div>`
    : `<div class="empty">Tap a line on the page to write on it.</div><div id="warnings"></div><div class="stack" id="assignedwrap"></div>`;

  if (line) {
    const ta = $("#linetext");
    ta.addEventListener("input", () => {
      store.assignments[line.id] = ta.value;
      persist(); refreshRuleClasses(); schedulePreview();
      $("#clearline").disabled = !ta.value;
    });
    $("#prevline").onclick = () => selectLine(p.lines[idx - 1].id, true);
    $("#nextline").onclick = () => selectLine(p.lines[idx + 1].id, true);
    $("#clearline").onclick = () => { delete store.assignments[line.id]; persist(); refreshRuleClasses(); renderText(); schedulePreview(0); };
    $("#importtxt").onchange = async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const text = (await f.text()).replace(/\s+/g, " ").trim();
      store.assignments[line.id] = text; persist(); refreshRuleClasses(); renderText(); schedulePreview(0);
      toast(`Imported ${text.split(" ").length} words from ${f.name}`);
    };
    $("#skipused").onchange = (e) => { store.skipUsed = e.target.checked; persist(); schedulePreview(0); };
  }
  renderWarnings();
}

function renderAssigned() {
  const wrap = $("#assignedwrap");
  if (!wrap) return;
  const placed = store.preview?.lines || [];
  if (!placed.length) { wrap.innerHTML = ""; return; }
  wrap.innerHTML = `
    <div class="row"><h3 class="grow">On this page</h3><button class="btn ghost small" id="clearpage">Clear page</button></div>
    <ul class="assigned-list">${placed.map((l) => `
      <li><button data-id="${l.line_id}" aria-current="${l.line_id === store.selected}">
        <b>${lineNum(l.line_id)}</b><span>${l.continued ? "… " : ""}${esc(l.text)}</span></button></li>`).join("")}
    </ul>
    ${store.preview.overflow ? `<p class="small" style="color:var(--danger)">Not written: “${esc(store.preview.overflow.slice(0, 120))}${store.preview.overflow.length > 120 ? "…" : ""}”</p>` : ""}`;
  $(".assigned-list", wrap).onclick = (e) => {
    const b = e.target.closest("button[data-id]");
    if (!b) return;
    const id = b.dataset.id;
    // A continued line belongs to the line its text started on.
    const own = store.assignments[id]?.trim() ? id : findOwner(id);
    selectLine(own);
  };
  $("#clearpage").onclick = () => {
    if (!confirm("Clear all text on this page?")) return;
    store.assignments = {}; persist(); refreshRuleClasses(); renderText(); schedulePreview(0);
  };
}

function findOwner(id) {
  const lines = store.page.lines.map((l) => l.id);
  for (let i = lines.indexOf(id); i >= 0; i--) if (store.assignments[lines[i]]?.trim()) return lines[i];
  return id;
}

// ---- Style tab
const SLIDERS = [
  ["x_height_mm", "Letter size", 1.6, 4.5, 0.1, (v) => `${v.toFixed(1)} mm`],
  ["slant_deg", "Slant", -10, 30, 1, (v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}°`],
  ["neatness", "Neatness", 0, 1, 0.05, (v) => (v < 0.35 ? "Rushed" : v < 0.7 ? "Everyday" : "Careful") + ` · ${Math.round(v * 100)}`],
  ["letter_spacing", "Letter spacing", 0.7, 1.4, 0.01, (v) => `${Math.round(v * 100)}%`],
  ["word_spacing", "Word spacing", 0.6, 1.8, 0.05, (v) => `${Math.round(v * 100)}%`],
];

function renderStyle() {
  const s = store.style;
  $("#tabbody").innerHTML = `
    <div class="stack">
      <h3>Profile</h3>
      <div class="seg" id="profiles">${store.presets.map((p) => `<button data-id="${p.id}" aria-pressed="${store.profile === p.id}">${esc(p.name)}</button>`).join("")}</div>
    </div>
    ${SLIDERS.map(([k, label, min, max, step, fmt]) => `
      <div class="slider">
        <label for="s-${k}">${label}</label><output id="o-${k}">${fmt(s[k])}</output>
        <input type="range" id="s-${k}" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${s[k]}">
      </div>`).join("")}
    <button class="btn" id="reseed">${ICON.shuffle} New variation</button>
    <p class="small muted">Same text, same settings and same variation always come out the same. “New variation” rewrites it a little differently, as you would.</p>
    ${store.status?.model_simulated !== false ? '<div class="notice">This is placeholder scribble until the handwriting model is trained on your samples (Capture).</div>' : ""}`;
  $("#profiles").onclick = (e) => {
    const b = e.target.closest("button[data-id]");
    if (!b) return;
    const p = store.presets.find((x) => x.id === b.dataset.id);
    store.style = { ...p.style, seed: store.style.seed };
    store.profile = p.id; persist(); renderStyle(); schedulePreview(0);
  };
  $$("input[type=range]", $("#tabbody")).forEach((inp) => inp.addEventListener("input", () => {
    const k = inp.dataset.k, v = parseFloat(inp.value);
    store.style = { ...store.style, [k]: v };
    store.profile = null;
    $(`#o-${k}`).textContent = SLIDERS.find((x) => x[0] === k)[5](v);
    $$("#profiles button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    persist(); schedulePreview(150);
  }));
  $("#reseed").onclick = () => { store.style = { ...store.style, seed: Math.floor(Math.random() * 1e6) }; persist(); schedulePreview(0); };
}

// ---- Plot tab
const fmtTime = (s) => (s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
const fmtLen = (mm) => (mm >= 1000 ? `${(mm / 1000).toFixed(1)} m` : `${Math.round(mm)} mm`);

function renderPlot() {
  const body = $("#tabbody");
  if (!body) return;
  const pv = store.preview, job = store.job;
  const active = job && ["running", "paused"].includes(job.state);
  const ink = pv?.ink_mm || 0;
  const secs = (ink / store.feedRate) * 60;

  if (active || (job && job.state === "done" && !job.trace_only) || (job && job.state === "stopped")) {
    const pct = Math.round(job.progress * 100);
    body.innerHTML = `
      <h2>${{ running: job.trace_only ? "Tracing the area" : "Writing", paused: "Paused", done: "Page written", stopped: "Stopped" }[job.state]}</h2>
      <div class="progress" aria-label="Progress"><span style="width:${pct}%"></span></div>
      <div class="facts">
        <div class="fact"><b>${pct}%</b><span>${fmtLen(job.drawn_mm)} of ${fmtLen(job.total_mm)}</span></div>
        <div class="fact"><b>${active ? fmtTime(job.eta_s) : "—"}</b><span>left at ${Math.round(job.feed_rate)} mm/min</span></div>
      </div>
      ${active ? `
        <div class="row">
          ${job.state === "running"
            ? `<button class="btn grow big" id="pause">${ICON.pause} Pause</button>`
            : `<button class="btn primary grow big" id="resume">${ICON.play} Resume</button>`}
          <button class="btn danger grow big" id="stop">${ICON.stop} Stop</button>
        </div>
        <p class="small muted">Pause lifts the pen and waits. Stop ends the job; what was written stays on the page.</p>`
        : `<a class="btn block" href="#/history">See it in History</a>`}`;
    if (active) {
      $("#pause") && ($("#pause").onclick = () => jobAction("pause"));
      $("#resume") && ($("#resume").onclick = () => jobAction("resume"));
      $("#stop").onclick = () => jobAction("stop");
    }
    return;
  }

  body.innerHTML = `
    <div class="facts">
      <div class="fact"><b>${pv?.lines.length || 0}</b><span>lines to write</span></div>
      <div class="fact"><b>${ink ? fmtTime(secs) : "—"}</b><span>${ink ? `${fmtLen(ink)} of ink` : "nothing yet"}</span></div>
    </div>
    <div class="slider">
      <label for="speed">Speed</label><output id="speedout">${store.feedRate} mm/min</output>
      <input type="range" id="speed" min="500" max="3000" step="100" value="${store.feedRate}">
      <span class="small muted" style="grid-column:1/-1">Slower is neater. Faster looks more natural but can blot.</span>
    </div>
    ${pv?.overflow ? '<div class="notice">Some text doesn’t fit and won’t be written. See the Text tab.</div>' : ""}
    <button class="btn block" id="trace" ${ink ? "" : "disabled"}>${ICON.trace} Trace the area first</button>
    <button class="btn primary block big" id="start" ${ink ? "" : "disabled"}>${ICON.play} Start writing</button>
    <div class="notice info">No robot connected. Jobs run on the simulator and the result is saved to History.</div>`;
  const sp = $("#speed");
  sp.oninput = () => {
    store.feedRate = +sp.value; persist();
    $("#speedout").textContent = `${store.feedRate} mm/min`;
    $(".fact:nth-child(2) b", body).textContent = ink ? fmtTime((ink / store.feedRate) * 60) : "—";
  };
  $("#trace").onclick = () => startJob(true);
  $("#start").onclick = () => startJob(false);
}

async function startJob(traceOnly) {
  try {
    store.job = await api("/api/jobs", { body: { ...writePayload(), feed_rate: store.feedRate, trace_only: traceOnly } });
    inkLengths = null;
    watchJob(); renderStatus(); onJobUpdate();
  } catch (e) { fail(e); }
}

async function jobAction(action) {
  try { store.job = await api(`/api/jobs/${store.job.id}/${action}`, { method: "POST" }); renderStatus(); onJobUpdate(); }
  catch (e) { fail(e); }
}

// ================================================================== SCAN

async function viewScan() {
  view.innerHTML = `
    <div class="pad">
      <div class="page-head">
        <div><h1>Choose a page</h1><p class="muted">Photograph the notebook page, or start from a template when you know the layout.</p></div>
      </div>
      <div class="scan-grid">
        <div class="card stack">
          <div class="row"><h2 class="grow">Photo</h2><span class="badge fake">Line detection simulated</span></div>
          <div class="camera" id="camera">
            <div class="hint">Lay the page flat in good light, with all four corners in view.</div>
          </div>
          <div class="row">
            <button class="btn grow" id="startcam">${ICON.camera} Use camera</button>
            <button class="btn primary grow" id="shoot" hidden>${ICON.camera} Take photo</button>
            <label class="btn grow file-btn">${ICON.upload} Upload photo<input type="file" id="upload" accept="image/*" capture="environment"></label>
          </div>
          <p class="small muted">For now the photo is used as-is and lines are guessed at 8 mm. Real corner and line detection comes next.</p>
        </div>
        <div class="stack">
          <h2>Templates</h2>
          <div class="tpl-grid" id="tpls"><div class="muted">Loading…</div></div>
        </div>
      </div>
    </div>`;

  const tpls = await api("/api/templates");
  $("#tpls").innerHTML = tpls.map((t) => `
    <button class="tpl" data-id="${t.id}">
      <img src="${t.thumb_url}" alt="" loading="lazy">
      <b>${esc(t.name)}</b><span>${esc(t.description)}</span>
    </button>`).join("");
  $("#tpls").onclick = async (e) => {
    const b = e.target.closest(".tpl");
    if (!b) return;
    try {
      setPage(await api("/api/pages/from-template", { body: { template_id: b.dataset.id } }));
      toast(`${b.querySelector("b").textContent} ready`);
      location.hash = "#/write";
    } catch (err) { fail(err); }
  };

  const usePhoto = async (blob, name = "photo.jpg") => {
    const form = new FormData();
    form.append("file", blob, name);
    try {
      setPage(await api("/api/pages/from-photo", { method: "POST", form }));
      toast("Photo scanned");
      location.hash = "#/write";
    } catch (err) { fail(err); }
  };
  $("#upload").onchange = (e) => e.target.files[0] && usePhoto(e.target.files[0], e.target.files[0].name);

  let stream = null;
  cleanup = () => stream?.getTracks().forEach((t) => t.stop());
  $("#startcam").onclick = async () => {
    if (!navigator.mediaDevices?.getUserMedia) return toast("This browser can’t open the camera here. Use Upload photo.", "error");
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 } }, audio: false });
    } catch (err) { return toast(`Camera unavailable: ${err.message}`, "error"); }
    $("#camera").innerHTML = '<video autoplay playsinline muted></video><div class="frame"></div>';
    $("#camera video").srcObject = stream;
    $("#startcam").hidden = true; $("#shoot").hidden = false;
  };
  $("#shoot").onclick = () => {
    const v = $("#camera video");
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((b) => usePhoto(b), "image/jpeg", 0.92);
  };
}

// ================================================================== CAPTURE

async function viewCapture() {
  const info = await api("/api/capture/prompts");
  let i = info.samples % info.prompts.length;
  let count = info.samples;
  view.innerHTML = `
    <div class="pad capture">
      <div class="page-head">
        <div><h1>Teach it your handwriting</h1><p class="muted">Write each sentence the way you normally would. Every sample trains the model.</p></div>
        <span class="badge accent num" id="count"></span>
      </div>
      <div class="progress"><span id="bar"></span></div>
      <p class="prompt" id="prompt"></p>
      <canvas class="pad-canvas" id="pad" aria-label="Writing area"></canvas>
      <div class="row">
        <button class="btn" id="undo">${ICON.undo} Undo stroke</button>
        <button class="btn ghost" id="clear">Clear</button>
        <button class="btn ghost" id="skip">Skip sentence</button>
        <span class="grow"></span>
        <button class="btn primary big" id="save" disabled>${ICON.check} Save and next</button>
      </div>
      <p class="small muted">A stylus or finger on a touchscreen gives the most natural samples. A mouse works for testing.</p>
    </div>`;

  const cv = $("#pad"), ctx = cv.getContext("2d");
  let strokes = [], cur = null, t0 = null;

  const showPrompt = () => {
    $("#prompt").textContent = info.prompts[i];
    $("#count").textContent = `${count} of ${info.goal} samples`;
    $("#bar").style.width = `${Math.min(100, (count / info.goal) * 100)}%`;
  };
  const resize = () => {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    redraw();
  };
  const redraw = () => {
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#c9d6ea"; ctx.lineWidth = 1;
    for (let y = h * 0.62; y < h; y += h * 0.5) { ctx.beginPath(); ctx.moveTo(12, y); ctx.lineTo(w - 12, y); ctx.stroke(); }
    ctx.strokeStyle = "#1b3a9a"; ctx.lineWidth = 2.2; ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (const s of strokes.concat(cur ? [cur] : [])) {
      ctx.beginPath();
      s.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (s.length === 1) ctx.lineTo(s[0][0] + 0.1, s[0][1]);
      ctx.stroke();
    }
    $("#save").disabled = !strokes.length;
  };
  const pt = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top, performance.now() - t0]; };

  cv.addEventListener("pointerdown", (e) => {
    e.preventDefault(); cv.setPointerCapture(e.pointerId);
    if (t0 === null) t0 = performance.now();
    cur = [pt(e)]; redraw();
  });
  cv.addEventListener("pointermove", (e) => {
    if (!cur) return;
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of evs.length ? evs : [e]) cur.push(pt(ev));
    redraw();
  });
  const end = () => { if (cur) { strokes.push(cur); cur = null; redraw(); } };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", end);

  const reset = () => { strokes = []; cur = null; t0 = null; redraw(); };
  $("#undo").onclick = () => { strokes.pop(); redraw(); };
  $("#clear").onclick = reset;
  $("#skip").onclick = () => { i = (i + 1) % info.prompts.length; reset(); showPrompt(); };
  $("#save").onclick = async () => {
    const points = [], times = [];
    for (const s of strokes) s.forEach(([x, y, t], k) => { points.push([+x.toFixed(2), +y.toFixed(2), k === s.length - 1 ? 1 : 0]); times.push(+t.toFixed(1)); });
    try {
      const r = await api("/api/capture/samples", { body: { prompt: info.prompts[i], points, times_ms: times, canvas_px: [cv.clientWidth, cv.clientHeight] } });
      count = r.samples; i = (i + 1) % info.prompts.length;
      toast(`Saved sample ${count}`);
      reset(); showPrompt();
    } catch (e) { fail(e); }
  };

  const ro = new ResizeObserver(resize);
  ro.observe(cv);
  cleanup = () => ro.disconnect();
  showPrompt();
}

// ================================================================== HISTORY

async function viewHistory() {
  const jobs = await api("/api/jobs");
  const fmt = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
  view.innerHTML = `
    <div class="pad">
      <div class="page-head"><div><h1>History</h1><p class="muted">Every page the robot has written, newest first.</p></div></div>
      ${jobs.length ? `<div class="jobs">${jobs.map((j) => `
        <article class="card job">
          ${j.image_url ? `<a href="${j.image_url}" target="_blank" rel="noopener"><img src="${j.image_url}" alt="Written page: ${esc(j.text_preview)}" loading="lazy"></a>` : ""}
          <div class="row"><h3 class="grow">${esc(j.label)}</h3><span class="badge ${j.state === "done" ? "ok" : "fake"}">${j.state === "done" ? "Finished" : "Stopped"}</span></div>
          <p class="text">${esc(j.text_preview) || '<span class="muted">—</span>'}</p>
          <p class="small muted num">${fmt.format(new Date(j.created))} · ${fmtLen(j.total_mm)} · ${Math.round(j.feed_rate)} mm/min</p>
        </article>`).join("")}</div>`
      : `<div class="card empty">Nothing written yet. Pages you write show up here.</div>`}
    </div>`;
}

// ================================================================== CALIBRATE

async function viewCalibrate() {
  const settings = await api("/api/settings");
  let measured = null, gridJob = null;
  view.innerHTML = `
    <div class="pad">
      <div class="page-head">
        <div><h1>Calibrate</h1><p class="muted">Make sure a millimetre on screen is a millimetre on paper.</p></div>
        <span class="badge fake">Measurements simulated</span>
      </div>
      <div class="steps">
        <section class="card step">
          <div class="row"><span class="step-n">1</span><h2>Plot a test grid</h2></div>
          <p class="muted">Put scrap A4 paper in. The robot draws 20 mm squares.</p>
          <div id="gridout"></div>
          <button class="btn primary" id="grid">${ICON.play} Plot test grid</button>
        </section>
        <section class="card step">
          <div class="row"><span class="step-n">2</span><h2>Photograph it</h2></div>
          <p class="muted">Take a straight-on photo of the grid, without moving the paper.</p>
          <label class="btn file-btn">${ICON.upload} Upload grid photo<input type="file" id="gridphoto" accept="image/*" capture="environment"></label>
          <button class="btn ghost" id="measure">Measure without a photo</button>
        </section>
        <section class="card step">
          <div class="row"><span class="step-n">3</span><h2>Apply the correction</h2></div>
          <dl class="kv" id="calvals"></dl>
          <button class="btn primary" id="apply" disabled>${ICON.check} Apply correction</button>
        </section>
      </div>

      <section class="card stack" style="margin-top:18px">
        <div class="row"><h2 class="grow">Machine</h2><span class="badge">Simulator</span></div>
        <div class="settings">
          <div class="field"><label for="pen">Pen width (mm)</label><input type="number" id="pen" min="0.1" max="2" step="0.05" value="${settings.pen_width_mm}"></div>
          <div class="field"><label for="feed">Default speed (mm/min)</label><input type="number" id="feed" min="200" max="6000" step="100" value="${settings.feed_rate}"></div>
          <div class="field"><label for="speedup">Simulator speed-up</label><input type="number" id="speedup" min="1" max="1000" step="1" value="${settings.sim_speedup}"></div>
          <div class="field"><label for="backend">Robot</label><select id="backend"><option>Simulator</option><option disabled>GRBL over USB (when the hardware arrives)</option></select></div>
        </div>
        <div class="row"><button class="btn primary" id="savesettings">Save settings</button></div>
      </section>
    </div>`;

  const showVals = (c, label) => {
    $("#calvals").innerHTML = `
      <dt>${label}</dt><dd></dd>
      <dt>Skew</dt><dd>${c.skew_deg.toFixed(2)}°</dd>
      <dt>Offset X</dt><dd>${c.offset_x_mm.toFixed(2)} mm</dd>
      <dt>Offset Y</dt><dd>${c.offset_y_mm.toFixed(2)} mm</dd>
      <dt>Scale</dt><dd>${(c.scale * 100).toFixed(2)}%</dd>`;
  };
  showVals(settings.calibration, "Current");

  const measure = async (file) => {
    const form = new FormData();
    if (file) form.append("file", file);
    try {
      measured = await api("/api/calibration/measure", { method: "POST", form });
      showVals(measured, "Measured");
      $("#apply").disabled = false;
    } catch (e) { fail(e); }
  };
  $("#gridphoto").onchange = (e) => e.target.files[0] && measure(e.target.files[0]);
  $("#measure").onclick = () => measure(null);
  $("#apply").onclick = async () => {
    try {
      const s = await api("/api/calibration/apply", { body: measured });
      showVals(s.calibration, "Current"); $("#apply").disabled = true; toast("Correction saved");
    } catch (e) { fail(e); }
  };

  $("#grid").onclick = async () => {
    try {
      gridJob = await api("/api/calibration/grid", { method: "POST" });
      store.job = gridJob; watchJob(); renderStatus();
    } catch (e) { fail(e); }
  };
  onJobUpdate = () => {
    const j = store.job;
    if (!j || !gridJob || j.id !== gridJob.id) return;
    const pct = Math.round(j.progress * 100);
    $("#gridout").innerHTML = j.image_url
      ? `<img class="grid-img" src="${j.image_url}" alt="Plotted test grid">`
      : `<div class="progress"><span style="width:${pct}%"></span></div><p class="small muted num">${pct}% drawn</p>`;
    $("#grid").disabled = ["running", "paused"].includes(j.state);
  };

  $("#savesettings").onclick = async () => {
    try {
      const s = await api("/api/settings", { method: "PUT", body: { pen_width_mm: +$("#pen").value, feed_rate: +$("#feed").value, sim_speedup: +$("#speedup").value } });
      store.feedRate = s.feed_rate; persist(); toast("Settings saved");
    } catch (e) { fail(e); }
  };
}

// ------------------------------------------------------------------ boot

pollStatus();
route();
