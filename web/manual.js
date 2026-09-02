let map, draw;
let currentJobId = null;
let samples = []; // [{id, class_name, lon, lat, polygon, thumbnail_url}]
let editingSampleId = null; // sample currently pulled into `draw` for editing, or null
let editingFeatureId = null; // the id mapbox-gl-draw assigned that feature
let pickingValidationOrigin = false;
let pickedOrigin = null; // {lon, lat} chosen via "Pick on Map", or null if not set yet
let validationCandidates = []; // last validation run's results
let knownClassNames = new Set();

const classSelect = document.getElementById("class-select");
const addClassToggleBtn = document.getElementById("add-class-toggle-btn");
const addClassPanel = document.getElementById("add-class-panel");
const addClassTypeSelect = document.getElementById("add-class-type-select");
const addClassParentLabel = document.getElementById("add-class-parent-label");
const addClassParentSelect = document.getElementById("add-class-parent-select");
const addClassNameInput = document.getElementById("add-class-name-input");
const addClassCreateBtn = document.getElementById("add-class-create-btn");

const tabBtnSamples = document.getElementById("tab-btn-samples");
const tabBtnValidation = document.getElementById("tab-btn-validation");
const tabBtnTraining = document.getElementById("tab-btn-training");
const tabBtnGraph = document.getElementById("tab-btn-graph");
const samplesTab = document.getElementById("samples-tab");
const validationTab = document.getElementById("validation-tab");
const trainingTab = document.getElementById("training-tab");
const graphTab = document.getElementById("graph-tab");
const trainingTreeEl = document.getElementById("training-tree");
const trainingEpochsInput = document.getElementById("training-epochs-input");
const trainingPatienceInput = document.getElementById("training-patience-input");
const trainingBaseModelInput = document.getElementById("training-base-model-input");

const graphParentHeaderEl = document.getElementById("graph-parent-header");
const graphDiagramEl = document.getElementById("graph-diagram");
const graphNodeEditor = document.getElementById("graph-node-editor");
const graphNodeEditorTitle = document.getElementById("graph-node-editor-title");
const graphNodeDependencyEl = document.getElementById("graph-node-dependency");
const graphNodeMinPieceInput = document.getElementById("graph-node-min-piece-input");
const graphNodeMaxPieceInput = document.getElementById("graph-node-max-piece-input");
const graphNodeSaveBtn = document.getElementById("graph-node-save-btn");
const graphEdgesListEl = document.getElementById("graph-edges-list");
const graphAddEdgeBtn = document.getElementById("graph-add-edge-btn");
const graphEdgeEditor = document.getElementById("graph-edge-editor");
const graphEdgeFromSelect = document.getElementById("graph-edge-from-select");
const graphEdgeToSelect = document.getElementById("graph-edge-to-select");
const graphEdgeMinDistInput = document.getElementById("graph-edge-min-dist-input");
const graphEdgeMaxDistInput = document.getElementById("graph-edge-max-dist-input");
const graphEdgeBoostInput = document.getElementById("graph-edge-boost-input");
const graphEdgeSaveBtn = document.getElementById("graph-edge-save-btn");
const graphEdgeCancelBtn = document.getElementById("graph-edge-cancel-btn");
const graphStatusEl = document.getElementById("graph-status");

mermaid.initialize({ startOnLoad: false, securityLevel: "loose" }); // "loose" is required for
// the click-a-node-to-edit-it callbacks below to actually fire -- default "strict" sandboxes them

const samplesListEl = document.getElementById("samples-list");
const generatePackageBtn = document.getElementById("generate-package-btn");
const generatePackageProgressEl = document.getElementById("generate-package-progress");
const generatePackageStatusEl = document.getElementById("generate-package-status");
const includeLatestCheckbox = document.getElementById("include-latest-checkbox");

const openValidationModalBtn = document.getElementById("open-validation-modal-btn");
const abortValidationBtn = document.getElementById("abort-validation-btn");
const validationStatusEl = document.getElementById("validation-status");
const validationResultsEl = document.getElementById("validation-results");

const validationModal = document.getElementById("validation-modal");
const validationZoomInput = document.getElementById("validation-zoom-input");
const validationThresholdInput = document.getElementById("validation-threshold-input");
const validationMaxFetchesInput = document.getElementById("validation-max-fetches-input");
const validationPositionDisplay = document.getElementById("validation-position-display");
const validationPickPositionBtn = document.getElementById("validation-pick-position-btn");
const validationModalCancel = document.getElementById("validation-modal-cancel");
const validationModalRun = document.getElementById("validation-modal-run");

const warningModal = document.getElementById("warning-modal");
const warningModalText = document.getElementById("warning-modal-text");
const warningModalOk = document.getElementById("warning-modal-ok");

function showWarning(text) {
  warningModalText.textContent = text;
  warningModal.style.display = "flex";
}

const lightboxModal = document.getElementById("lightbox-modal");
const lightboxBox = document.getElementById("lightbox-box");

function openLightbox(candidate) {
  lightboxBox.innerHTML = "";
  const closeBtn = document.createElement("button");
  closeBtn.className = "lightbox-close";
  closeBtn.textContent = "✕";
  closeBtn.title = "Close";
  closeBtn.addEventListener("click", closeLightbox);
  lightboxBox.appendChild(closeBtn);

  const img = document.createElement("img");
  img.src = candidate.thumbnail_url;
  lightboxBox.appendChild(img);

  lightboxBox.insertAdjacentHTML("beforeend", labelOverlaySvg(candidate.label_polygon));

  const sim = document.createElement("span");
  sim.className = "sim";
  sim.textContent = candidate.similarity.toFixed(2);
  lightboxBox.appendChild(sim);

  lightboxModal.style.display = "flex";
}

function closeLightbox() {
  lightboxModal.style.display = "none";
}

lightboxModal.addEventListener("click", (e) => {
  if (e.target === lightboxModal) closeLightbox(); // click on the backdrop, not the image itself
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && lightboxModal.style.display !== "none") closeLightbox();
});

function currentClassName() {
  return classSelect.value;
}

// The add-class panel is a form with dependent fields: the parent picker only makes sense (and
// only shows) once "New sub-class" is picked, and "Create Class" only enables once every field
// relevant to the current type is actually filled in -- prevents submitting a sub-class with no
// parent chosen, which is what silently produced a confusing top-level class before.
function updateAddClassPanelState() {
  const isSub = addClassTypeSelect.value === "sub";
  addClassParentLabel.style.display = isSub ? "block" : "none";
  addClassParentSelect.style.display = isSub ? "block" : "none";
  updateAddClassCreateEnabled();
}

function updateAddClassCreateEnabled() {
  const isSub = addClassTypeSelect.value === "sub";
  const nameOk = !!addClassNameInput.value.trim();
  const parentOk = !isSub || !!addClassParentSelect.value;
  addClassCreateBtn.disabled = !nameOk || !parentOk;
}

function openAddClassPanel() {
  addClassPanel.style.display = "block";
  addClassTypeSelect.value = "parent";
  addClassParentSelect.value = "";
  addClassNameInput.value = "";
  updateAddClassPanelState();
  addClassNameInput.focus();
}

function closeAddClassPanel() {
  addClassPanel.style.display = "none";
}

async function createNewClass() {
  const name = addClassNameInput.value.trim();
  const isSub = addClassTypeSelect.value === "sub";
  const parent = isSub ? addClassParentSelect.value : null;
  if (!name || (isSub && !parent)) return;
  const fullName = parent ? `${parent}/${name}` : name;
  if (knownClassNames.has(fullName)) {
    classSelect.value = fullName;
    closeAddClassPanel();
    await loadSamples();
    if (trainingTab.style.display !== "none") loadTrainingPanel();
    if (graphTab.style.display !== "none") loadGraphTab();
    return;
  }
  const res = await fetch("/api/classes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, parent }),
  });
  if (!res.ok) {
    alert("Could not create class: " + (await res.text()));
    return;
  }
  await loadClasses();
  classSelect.value = fullName;
  closeAddClassPanel();
  await loadSamples();
  if (trainingTab.style.display !== "none") loadTrainingPanel();
}

// ---------- geometry helpers (same math as the main app's app.js, kept separate since the two
// pages' interaction models diverge enough that sharing a module isn't worth the wiring) ----------

function polygonCentroid(ring) {
  const pts = ring.slice(0, -1); // last point repeats the first
  const lon = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const lat = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return { lon, lat };
}

function polygonBbox(ring) {
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  return { west: Math.min(...lons), east: Math.max(...lons), south: Math.min(...lats), north: Math.max(...lats) };
}

function labelOverlaySvg(labelPolygons) {
  if (!labelPolygons || !labelPolygons.length) return "";
  const shapes = labelPolygons
    .filter((poly) => poly && poly.length)
    .map((poly) => `<polygon points="${poly.map((p) => p.join(",")).join(" ")}" />`)
    .join("");
  return `<svg class="label-overlay" viewBox="0 0 1 1" preserveAspectRatio="none">${shapes}</svg>`;
}

async function pollJob(jobId, { onProgress, intervalMs = 1000 } = {}) {
  while (true) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (job.status !== "running") return job;
    if (onProgress) onProgress(job.progress);
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

// ---------- map + draw setup ----------

async function loadConfig() {
  const res = await fetch("/api/config");
  const { mapbox_token } = await res.json();

  const style = {
    version: 8,
    glyphs: "https://fonts.openmaptiles.org/{fontstack}/{range}.pbf",
    sources: {
      satellite: {
        type: "raster",
        tiles: [`https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg90?access_token=${mapbox_token}`],
        tileSize: 256,
        attribution: "© Mapbox",
      },
    },
    layers: [{ id: "satellite", type: "raster", source: "satellite" }],
  };

  map = new maplibregl.Map({ container: "map", style, center: [0, 20], zoom: 2 });
  map.addControl(new maplibregl.NavigationControl());

  draw = new MapboxDraw({
    displayControlsDefault: false,
    controls: { polygon: true, trash: true },
  });
  map.addControl(draw);

  map.on("load", () => {
    // Static, non-interactive rendering of every saved sample except the one currently pulled
    // into `draw` for editing -- kept out of MapboxDraw entirely so its own click-driven
    // simple_select/direct_select state machine never has more than one feature to worry about.
    map.addSource("samples-source", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({
      id: "samples-fill", type: "fill", source: "samples-source",
      paint: { "fill-color": "#3b82f6", "fill-opacity": 0.2 },
    });
    map.addLayer({
      id: "samples-line", type: "line", source: "samples-source",
      paint: { "line-color": "#3b82f6", "line-width": 2 },
    });

    // mapbox-gl-draw's draw_polygon mode only ever renders a vertex marker for the first and
    // most-recently-placed point (confirmed from the library's own source) -- this layer fills
    // in a dot for every already-placed vertex while actively drawing a new shape.
    map.addSource("vertex-dots-source", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({
      id: "vertex-dots", type: "circle", source: "vertex-dots-source",
      paint: { "circle-radius": 4, "circle-color": "#3b82f6", "circle-stroke-width": 1, "circle-stroke-color": "#fff" },
    });

    refreshSamplesLayer();
  });

  map.on("draw.create", (e) => handleNewShape(e.features[0]));

  map.on("draw.modechange", (e) => {
    map.getCanvas().style.cursor = e.mode === "draw_polygon" ? "crosshair" : "";
    if (e.mode !== "draw_polygon") clearVertexDots();
  });
  map.on("draw.render", () => {
    if (draw.getMode() === "draw_polygon") updateVertexDots();
  });

  // Double-click a static (blue) sample shape -> pull it into `draw` for editing (orange, full
  // vertex drag/add/remove support, all native to direct_select -- confirmed working earlier).
  map.on("dblclick", "samples-fill", (e) => {
    e.preventDefault(); // stop MapboxDraw's own dblclick-finishes-a-shape handling from firing
    const sampleId = e.features[0].properties.sampleId;
    startEditingSample(sampleId);
  });

  // Finishing an edit: mapbox-gl-draw fires an empty selectionchange on deselect (click
  // elsewhere). Only acts when we're actually mid-edit, so a fresh draw's own auto-select
  // doesn't get mistaken for "done editing".
  map.on("draw.selectionchange", (e) => {
    if (editingSampleId && e.features.length === 0) finishEditingSample();
  });

  map.on("click", (e) => {
    if (!pickingValidationOrigin) return;
    pickingValidationOrigin = false;
    map.getCanvas().style.cursor = "";
    pickedOrigin = { lon: e.lngLat.lng, lat: e.lngLat.lat };
    validationPositionDisplay.textContent = `${pickedOrigin.lat.toFixed(5)}, ${pickedOrigin.lon.toFixed(5)}`;
    validationModalRun.disabled = false;
    validationModal.style.display = "flex";
  });
}

function clearVertexDots() {
  const source = map.getSource("vertex-dots-source");
  if (source) source.setData({ type: "FeatureCollection", features: [] });
}

function updateVertexDots() {
  // `draw` only ever holds the one feature currently being drawn (or edited) in this app's
  // design -- draw.getAll()'s features don't carry the render-only "mode"/"active" tags used
  // internally by mapbox-gl-draw (those exist only in its raw GeoJSON render source), so there's
  // nothing to filter by; taking the first (only) feature is correct here.
  const inProgress = draw.getAll().features[0];
  const source = map.getSource("vertex-dots-source");
  if (!inProgress || !source) {
    clearVertexDots();
    return;
  }
  const ring = inProgress.geometry.coordinates[0] || [];
  const features = ring.map((coord) => ({ type: "Feature", geometry: { type: "Point", coordinates: coord }, properties: {} }));
  source.setData({ type: "FeatureCollection", features });
}

function refreshSamplesLayer() {
  const source = map.getSource("samples-source");
  if (!source) return;
  const features = samples
    .filter((s) => s.id !== editingSampleId && s.polygon)
    .map((s) => ({
      type: "Feature",
      properties: { sampleId: s.id },
      geometry: { type: "Polygon", coordinates: [s.polygon] },
    }));
  source.setData({ type: "FeatureCollection", features });
}

// ---------- drawing a new sample ----------

async function handleNewShape(feature) {
  const className = currentClassName();
  if (!className) {
    draw.delete(feature.id);
    showWarning("Pick or name a class first (top of the sidebar) before drawing a sample.");
    return;
  }

  const ring = feature.geometry.coordinates[0];
  const bbox = polygonBbox(ring);
  const zoom = Math.round(map.getZoom());
  const { lon, lat } = polygonCentroid(ring);
  draw.delete(feature.id); // saved samples live on the static layer, not in `draw`

  const res = await fetch("/api/manual/samples", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ class_name: className, lat, lon, zoom, ...bbox, polygon: ring }),
  });
  if (!res.ok) {
    showWarning("Failed to save sample: " + (await res.text()));
    return;
  }
  const sample = await res.json();
  samples.push(sample);
  refreshSamplesLayer();
  renderSamplesList();
}

// ---------- editing an existing sample ----------

function startEditingSample(sampleId) {
  if (editingSampleId) return; // already editing something -- finish that first
  const sample = samples.find((s) => s.id === sampleId);
  if (!sample) return;
  editingSampleId = sampleId;
  const [id] = draw.add({ type: "Feature", geometry: { type: "Polygon", coordinates: [sample.polygon] }, properties: {} });
  editingFeatureId = id;
  refreshSamplesLayer(); // hide the static copy while its `draw` copy is being edited
  draw.changeMode("direct_select", { featureId: editingFeatureId });
}

async function finishEditingSample() {
  const sampleId = editingSampleId;
  const featureId = editingFeatureId;
  const feature = draw.get(featureId);
  editingSampleId = null;
  editingFeatureId = null;
  if (!feature) {
    refreshSamplesLayer();
    return;
  }
  draw.delete(featureId);

  const ring = feature.geometry.coordinates[0];
  const res = await fetch(`/api/manual/samples/${sampleId}?class_name=${encodeURIComponent(currentClassName())}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ polygon: ring }),
  });
  if (res.ok) {
    const updated = await res.json();
    const i = samples.findIndex((s) => s.id === sampleId);
    if (i !== -1) samples[i] = updated;
  } else {
    showWarning("Failed to save edit: " + (await res.text()));
  }
  refreshSamplesLayer();
  renderSamplesList();
}

// ---------- samples tab ----------

let samplesRequestId = 0; // guards against a slow/stale fetch for a previously-selected class
// overwriting the currently-selected class's freshly-loaded data if responses arrive out of order

async function loadSamples() {
  const className = currentClassName();
  const requestId = ++samplesRequestId;

  // clear immediately so switching classes never shows a stale mix, even before the fetch below resolves
  samples = [];
  editingSampleId = null;
  editingFeatureId = null;
  refreshSamplesLayer();
  renderSamplesList();
  if (!className) return;

  const res = await fetch(`/api/manual/samples?class_name=${encodeURIComponent(className)}`);
  const data = await res.json();
  if (requestId !== samplesRequestId) return; // a newer loadSamples() call has since superseded this one
  samples = data.samples || [];
  refreshSamplesLayer();
  renderSamplesList();
}

function renderSamplesList() {
  samplesListEl.innerHTML = "";
  for (const s of samples) {
    const row = document.createElement("div");
    row.className = "sample-row";
    const img = document.createElement("img");
    img.src = s.thumbnail_url;
    row.appendChild(img);
    const delBtn = document.createElement("button");
    delBtn.textContent = "✕";
    delBtn.title = "Delete this sample";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/api/manual/samples/${s.id}?class_name=${encodeURIComponent(currentClassName())}`, { method: "DELETE" });
      samples = samples.filter((x) => x.id !== s.id);
      refreshSamplesLayer();
      renderSamplesList();
    });
    row.appendChild(delBtn);
    row.addEventListener("click", () => map.flyTo({ center: [s.lon, s.lat], zoom: 18 }));
    samplesListEl.appendChild(row);
  }
}

async function generatePackage() {
  const className = currentClassName();
  if (!className) return;
  generatePackageBtn.disabled = true;
  generatePackageProgressEl.value = 0;
  generatePackageProgressEl.style.display = "block";
  generatePackageStatusEl.textContent = "Starting...";
  try {
    const res = await fetch("/api/manual/generate_package", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_name: className, include_latest: includeLatestCheckbox.checked }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();

    const job = await pollJob(job_id, {
      onProgress: (p) => {
        generatePackageProgressEl.value = p.percent || 0;
        generatePackageStatusEl.textContent = p.detail ? `${p.step} -- ${p.detail}` : (p.step || "Working...");
      },
    });
    if (job.status === "error") throw new Error(job.error);

    const data = job.result;
    const s3Note = data.s3_configured
      ? (data.s3_key ? "uploaded to S3" : "S3 upload FAILED, check logs")
      : "local only";
    const mergeNote = data.merge && data.merge.added_from_remote > 0
      ? `merged ${data.merge.added_from_remote} from S3, `
      : "";
    generatePackageProgressEl.value = 100;
    generatePackageStatusEl.textContent =
      `Done -- ${mergeNote}seg ${data.segmentation.train}/${data.segmentation.val}, ` +
      `obb ${data.obb.train}/${data.obb.val} (train/val), ${s3Note}.`;
  } catch (err) {
    generatePackageStatusEl.textContent = "Error: " + err.message;
  } finally {
    generatePackageBtn.disabled = false;
    generatePackageProgressEl.style.display = "none";
  }
}

// ---------- tabs ----------

function switchTab(tab) {
  samplesTab.style.display = tab === "samples" ? "block" : "none";
  validationTab.style.display = tab === "validation" ? "block" : "none";
  trainingTab.style.display = tab === "training" ? "block" : "none";
  graphTab.style.display = tab === "graph" ? "block" : "none";
  tabBtnSamples.classList.toggle("active", tab === "samples");
  tabBtnValidation.classList.toggle("active", tab === "validation");
  tabBtnTraining.classList.toggle("active", tab === "training");
  tabBtnGraph.classList.toggle("active", tab === "graph");
  if (tab === "training") loadTrainingPanel();
  if (tab === "graph") loadGraphTab();
}

// ---------- training tab ----------
// Scoped to whatever class is currently picked in the Class dropdown at the top of the sidebar
// -- that selection is the single source of truth for which class this panel acts on, so it
// isn't repeated again as a label down here, and no other class's row is shown alongside it.

async function loadTrainingPanel() {
  const className = currentClassName();
  trainingTreeEl.innerHTML = "";
  if (!className) return;

  const [classesRes, activeRes] = await Promise.all([
    fetch("/api/classes").then((r) => r.json()),
    fetch("/api/train/active").then((r) => r.json()),
  ]);
  const { classes, parents } = classesRes;
  const activeJobs = activeRes.jobs || {};
  const children = classes.filter((c) => parents[c] === className);

  const row = buildTrainingRow(className, children);
  trainingTreeEl.appendChild(row);

  if (activeJobs[className]) watchTrainingJob(row, activeJobs[className]);
}

function buildTrainingRow(className, children) {
  // Stacked, not a flex row: a fixed-width button sitting next to a flex-growing status span
  // risked the button's own `width:100%` (from the global `button` rule) fighting the status
  // span for space and squeezing its text down to nothing -- full-width, stacked lines can't do
  // that, and it matches every other button in this sidebar (Generate Package, Add new class).
  const wrap = document.createElement("div");
  wrap.className = "training-row-wrap";

  const btn = document.createElement("button");
  btn.className = "training-row-btn";
  btn.textContent = "Train";
  btn.addEventListener("click", () => startTraining(className, wrap));
  wrap.appendChild(btn);

  const progress = document.createElement("progress");
  progress.className = "training-row-progress";
  progress.max = 100;
  progress.value = 0;
  progress.style.visibility = "hidden"; // reserves its layout space even while idle, so it
  // appearing/disappearing never shifts anything around it (kept invisible, not display:none)
  wrap.appendChild(progress);

  const status = document.createElement("div");
  status.className = "training-row-status";
  wrap.appendChild(status);

  if (children.length) {
    const subLabel = document.createElement("label");
    subLabel.className = "training-row-subclass-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "training-row-include-subclasses";
    subLabel.appendChild(checkbox);
    subLabel.appendChild(document.createTextNode(
      ` include sub-class samples (${children.map((c) => c.split("/").pop()).join(", ")})`,
    ));
    wrap.appendChild(subLabel);
  }

  return wrap;
}

// Metric keys come back as e.g. "metrics/precision(B)" -- strip that down to "precision" and
// round to 2 decimals so a mid-training status line stays a glance-able one-liner.
function shortMetrics(prefix, metrics) {
  if (!metrics) return prefix;
  const parts = Object.entries(metrics).map(([k, v]) => {
    const name = k.split("/").pop().replace(/\(B\)$/, "");
    const val = typeof v === "number" ? v.toFixed(2) : v;
    return `${name}=${val}`;
  });
  return `${prefix} (${parts.join(", ")})`;
}

async function startTraining(className, wrap) {
  const btn = wrap.querySelector(".training-row-btn");
  const status = wrap.querySelector(".training-row-status");
  const includeSubclassesCheckbox = wrap.querySelector(".training-row-include-subclasses");
  btn.disabled = true;
  status.textContent = "Starting...";
  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        class_name: className,
        epochs: parseInt(trainingEpochsInput.value, 10) || 100,
        patience: parseInt(trainingPatienceInput.value, 10) || 30,
        base_model: trainingBaseModelInput.value.trim() || "yolo11n-obb.pt",
        include_subclasses: includeSubclassesCheckbox ? includeSubclassesCheckbox.checked : false,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    await watchTrainingJob(wrap, job_id);
  } catch (err) {
    status.textContent = "Error: " + err.message;
    btn.disabled = false;
  }
}

async function watchTrainingJob(row, jobId) {
  const btn = row.querySelector(".training-row-btn");
  const status = row.querySelector(".training-row-status");
  const progress = row.querySelector("progress");
  btn.disabled = true;
  progress.style.visibility = "visible";

  const job = await pollJob(jobId, {
    intervalMs: 3000,
    onProgress: (p) => {
      progress.value = p.percent || 0;
      status.textContent = shortMetrics(p.step, p.metrics);
    },
  });

  progress.style.visibility = "hidden";
  btn.disabled = false;
  if (job.status === "error") {
    status.textContent = "Error: " + job.error;
  } else if (job.status === "done") {
    const m = job.result && job.result.metrics && job.result.metrics.metrics;
    status.textContent = shortMetrics(`Saved ${job.result.version}`, m);
  }
}

// ---------- graph tab ----------
// Edits classes/<parent>/subclass_graph.json: nodes (a class's own piece-cutting min/max, edited
// by clicking the node in the Mermaid diagram) and edges (spatial-proximity confidence-boost
// rules between two sub-classes, edited via the plain list below the diagram -- Mermaid's click
// callback only fires on nodes, not edges, so editing an edge in the diagram itself isn't an
// option here).

let currentGraph = null; // {parent, available_nodes, nodes, edges} from the last GET/POST
let editingEdgeIndex = null; // index into currentGraph.edges being edited, or null when adding new

function graphNodeId(name) {
  return "n_" + name.replace(/[^a-zA-Z0-9_]/g, "_");
}

async function loadGraphTab() {
  const className = currentClassName();
  graphNodeEditor.style.display = "none";
  graphEdgeEditor.style.display = "none";
  graphStatusEl.textContent = "";
  if (!className) {
    graphDiagramEl.innerHTML = "";
    graphEdgesListEl.innerHTML = "";
    currentGraph = null;
    return;
  }
  const res = await fetch(`/api/subclass_graph?class_name=${encodeURIComponent(className)}`);
  currentGraph = await res.json();
  await renderGraphDiagram();
  renderEdgesList();
}

async function renderGraphDiagram() {
  const { parent, available_nodes, edges } = currentGraph;
  const subClasses = available_nodes.filter((n) => n !== parent);
  const selectedBare = currentClassName().split("/").pop(); // whichever node the top Class
  // dropdown currently points at -- highlighted here so the diagram reflects the real selection,
  // not a fixed "this one is the parent" styling that stays on regardless of what's picked.

  // The parent isn't a node in the Mermaid graph at all -- it doesn't have siblings to be laid
  // out against, so it doesn't need dagre's automatic ranking, and giving it permanent special
  // styling there made it look "highlighted/selected" even when it wasn't. It's just a plain
  // clickable header above a pure sibling graph instead.
  graphParentHeaderEl.textContent = `${parent} (parent)`;
  graphParentHeaderEl.onclick = () => onGraphNodeClick(parent);
  graphParentHeaderEl.classList.toggle("selected", selectedBare === parent);

  if (!subClasses.length) {
    graphDiagramEl.innerHTML = '<p class="hint">No sub-classes yet.</p>';
    return;
  }

  // graph LR (not TD): sub-classes are pure siblings with no parent node pulling on the layout,
  // so adding more of them grows the row sideways instead of stacking new ones underneath.
  const lines = [
    "graph LR",
    "classDef selectedNode fill:#e3ecfb,stroke:#1a73e8,stroke-width:2px;",
  ];
  for (const name of subClasses) {
    const cls = name === selectedBare ? ":::selectedNode" : "";
    const label = incomingEdgesFor(name).length ? name : `${name} (seed)`;
    lines.push(`  ${graphNodeId(name)}["${label}"]${cls}`);
  }
  for (const edge of edges) {
    const label = `${edge.min_distance_m ?? 0}-${edge.max_distance_m}m +${edge.boost}`;
    lines.push(`  ${graphNodeId(edge.from)} -->|"${label}"| ${graphNodeId(edge.to)}`);
  }
  for (const name of subClasses) {
    lines.push(`  click ${graphNodeId(name)} call onGraphNodeClick("${name}")`);
  }

  const { svg, bindFunctions } = await mermaid.render("graph-mermaid-" + Date.now(), lines.join("\n"));
  graphDiagramEl.innerHTML = svg;
  if (bindFunctions) bindFunctions(graphDiagramEl); // without this, "click ... call ..." above never fires
}

// Edges where `name` is the "to" -- i.e. edges whose max_distance_m/boost describe how close
// `name`'s own detections need to be to some other (anchor) sub-class to get boosted. A
// sub-class with none of these is a "seed": nothing it depends on, so no proximity check ever
// applies to it (though it can still be the anchor other sub-classes look for).
function incomingEdgesFor(name) {
  return (currentGraph.edges || []).filter((e) => e.to === name);
}

window.onGraphNodeClick = function (name) {
  const cfg = (currentGraph.nodes && currentGraph.nodes[name]) || {};
  graphNodeEditorTitle.textContent = `Piece size for "${name}"`;

  const isParent = name === currentGraph.parent;
  const incoming = isParent ? [] : incomingEdgesFor(name);
  if (isParent) {
    graphNodeDependencyEl.textContent = "";
  } else if (!incoming.length) {
    graphNodeDependencyEl.textContent = "Seed sub-class -- no proximity dependency.";
  } else {
    graphNodeDependencyEl.textContent = incoming
      .map((e) => `Depends on "${e.from}": ${e.min_distance_m ?? 0}-${e.max_distance_m}m, +${e.boost} (edit in Edges below)`)
      .join(" ");
  }

  graphNodeMinPieceInput.value = cfg.min_piece_m ?? "";
  graphNodeMaxPieceInput.value = cfg.max_piece_m ?? "";
  graphNodeEditor.dataset.node = name;
  graphNodeEditor.style.display = "block";
  graphEdgeEditor.style.display = "none";
};

function renderEdgesList() {
  graphEdgesListEl.innerHTML = "";
  if (!currentGraph.edges.length) {
    graphEdgesListEl.innerHTML = '<p class="hint">No edges yet.</p>';
  }
  currentGraph.edges.forEach((edge, i) => {
    const row = document.createElement("div");
    row.className = "graph-edge-row";

    const label = document.createElement("span");
    label.className = "graph-edge-label";
    label.textContent = `${edge.from} -> ${edge.to}  (${edge.min_distance_m ?? 0}-${edge.max_distance_m}m, +${edge.boost})`;
    row.appendChild(label);

    const editBtn = document.createElement("button");
    editBtn.className = "secondary";
    editBtn.textContent = "Edit";
    editBtn.addEventListener("click", () => openEdgeEditor(i));
    row.appendChild(editBtn);

    const removeBtn = document.createElement("button");
    removeBtn.className = "danger";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => removeEdge(i));
    row.appendChild(removeBtn);

    graphEdgesListEl.appendChild(row);
  });
}

function openEdgeEditor(index) {
  editingEdgeIndex = index;
  graphEdgeFromSelect.innerHTML = "";
  graphEdgeToSelect.innerHTML = "";
  // The parent itself is never a valid edge endpoint -- the proximity graph only relates
  // sibling sub-classes to each other, so it's left out of these dropdowns entirely rather than
  // being selectable and then rejected after the fact.
  const edgeableNodes = currentGraph.available_nodes.filter((n) => n !== currentGraph.parent);
  for (const name of edgeableNodes) {
    for (const sel of [graphEdgeFromSelect, graphEdgeToSelect]) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
  }
  if (index === null) {
    graphEdgeMinDistInput.value = 0;
    graphEdgeMaxDistInput.value = 5;
    graphEdgeBoostInput.value = 0.2;
  } else {
    const edge = currentGraph.edges[index];
    graphEdgeFromSelect.value = edge.from;
    graphEdgeToSelect.value = edge.to;
    graphEdgeMinDistInput.value = edge.min_distance_m ?? 0;
    graphEdgeMaxDistInput.value = edge.max_distance_m;
    graphEdgeBoostInput.value = edge.boost;
  }
  graphEdgeEditor.style.display = "block";
  graphNodeEditor.style.display = "none";
}

async function saveGraph() {
  const className = currentClassName();
  const res = await fetch("/api/subclass_graph", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ class_name: className, nodes: currentGraph.nodes, edges: currentGraph.edges }),
  });
  if (!res.ok) {
    graphStatusEl.textContent = "Error: " + (await res.text());
    return false;
  }
  const saved = await res.json();
  currentGraph.nodes = saved.nodes;
  currentGraph.edges = saved.edges;
  graphStatusEl.textContent = "Saved.";
  return true;
}

graphNodeSaveBtn.addEventListener("click", async () => {
  const name = graphNodeEditor.dataset.node;
  const minV = parseFloat(graphNodeMinPieceInput.value);
  const maxV = parseFloat(graphNodeMaxPieceInput.value);
  if (isNaN(minV) || isNaN(maxV)) {
    graphStatusEl.textContent = "Enter both min and max piece size.";
    return;
  }
  currentGraph.nodes[name] = { min_piece_m: minV, max_piece_m: maxV };
  if (await saveGraph()) {
    graphNodeEditor.style.display = "none";
    await renderGraphDiagram();
  }
});

graphAddEdgeBtn.addEventListener("click", () => openEdgeEditor(null));
graphEdgeCancelBtn.addEventListener("click", () => { graphEdgeEditor.style.display = "none"; });

graphEdgeSaveBtn.addEventListener("click", async () => {
  const edge = {
    from: graphEdgeFromSelect.value,
    to: graphEdgeToSelect.value,
    min_distance_m: graphEdgeMinDistInput.value.trim() === "" ? 0 : parseFloat(graphEdgeMinDistInput.value),
    max_distance_m: parseFloat(graphEdgeMaxDistInput.value),
    boost: parseFloat(graphEdgeBoostInput.value),
  };
  if (edge.from === edge.to) {
    graphStatusEl.textContent = "From and To must be different sub-classes.";
    return;
  }
  if (isNaN(edge.min_distance_m) || isNaN(edge.max_distance_m) || isNaN(edge.boost)) {
    graphStatusEl.textContent = "Min/max distance and boost must all be numbers.";
    return;
  }
  if (edge.min_distance_m > edge.max_distance_m) {
    graphStatusEl.textContent = "Min distance can't be greater than max distance.";
    return;
  }
  const dupIndex = currentGraph.edges.findIndex((e) => e.from === edge.from && e.to === edge.to);
  if (dupIndex !== -1 && dupIndex !== editingEdgeIndex) {
    graphStatusEl.textContent = `An edge from "${edge.from}" to "${edge.to}" already exists -- edit that one instead.`;
    return;
  }
  if (editingEdgeIndex === null) currentGraph.edges.push(edge);
  else currentGraph.edges[editingEdgeIndex] = edge;
  if (await saveGraph()) {
    graphEdgeEditor.style.display = "none";
    await renderGraphDiagram();
    renderEdgesList();
  }
});

async function removeEdge(index) {
  currentGraph.edges.splice(index, 1);
  if (await saveGraph()) {
    await renderGraphDiagram();
    renderEdgesList();
  }
}

// ---------- validation tab ----------

function openValidationModal() {
  const className = currentClassName();
  if (!className) {
    showWarning("Pick or name a class first.");
    return;
  }
  if (!samples.length) {
    showWarning("No samples yet for this class -- draw at least one before running validation.");
    return;
  }
  validationModal.style.display = "flex";
}

// Clicking "Pick on Map" hides the dialog so the map is fully clickable; the map's own click
// handler (see loadConfig) captures the point, then reopens this same dialog with the position
// filled in -- parameters you already set (zoom/threshold/tiles) are untouched throughout.
function startPickingPosition() {
  validationModal.style.display = "none";
  pickingValidationOrigin = true;
  map.getCanvas().style.cursor = "crosshair";
}

function runValidationFromModal() {
  if (!pickedOrigin) return;
  validationModal.style.display = "none";
  startValidation(pickedOrigin.lon, pickedOrigin.lat);
}

async function startValidation(lon, lat) {
  abortValidationBtn.style.display = "block";
  validationResultsEl.innerHTML = "";
  setValidationStatus("Starting validation...");

  try {
    const res = await fetch("/api/manual/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        class_name: currentClassName(), lat, lon,
        zoom: parseFloat(validationZoomInput.value) || 17,
        threshold: parseFloat(validationThresholdInput.value) || 0.75,
        max_fetches: parseInt(validationMaxFetchesInput.value, 10) || 300,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    currentJobId = job_id;

    const job = await pollJob(job_id, {
      onProgress: (p) => setValidationStatus(`Searching... ${p.fetched_count || 0} tile(s) checked, ${p.candidates_found || 0} found so far.`),
    });
    currentJobId = null;
    abortValidationBtn.style.display = "none";

    if (job.status === "error") throw new Error(job.error);
    validationCandidates = job.result.candidates;

    if (job.status === "aborted") {
      setValidationStatus(`Aborted: ${validationCandidates.length} candidate(s) found before stopping.`);
    } else if (!validationCandidates.length) {
      setValidationStatus(`No candidates found (checked ${job.result.fetched_count} tiles). Your samples may not generalize well from this location/threshold yet.`);
    } else {
      setValidationStatus(`${validationCandidates.length} candidate(s) found from ${job.result.exemplar_count} exemplar(s), ${job.result.fetched_count} tile(s) checked.`);
    }
    renderValidationResults();
  } catch (err) {
    setValidationStatus("Error: " + err.message);
    abortValidationBtn.style.display = "none";
  }
}

function setValidationStatus(text) {
  validationStatusEl.textContent = text || "";
}

async function abortValidation() {
  if (!currentJobId) return;
  abortValidationBtn.disabled = true;
  try {
    await fetch(`/api/jobs/${currentJobId}/abort`, { method: "POST" });
  } finally {
    abortValidationBtn.disabled = false;
  }
}

function renderValidationResults() {
  validationResultsEl.innerHTML = "";
  for (const c of validationCandidates) {
    const wrap = document.createElement("div");
    wrap.className = "validation-result";

    const thumb = document.createElement("div");
    thumb.className = "thumb";
    const overlay = labelOverlaySvg(c.label_polygon);
    thumb.innerHTML = `<img src="${c.thumbnail_url}" />${overlay}<span class="sim">${c.similarity.toFixed(2)}</span>`;
    thumb.addEventListener("click", () => openLightbox(c));
    wrap.appendChild(thumb);

    const promoteBtn = document.createElement("button");
    promoteBtn.textContent = "Add to Samples";
    promoteBtn.addEventListener("click", async () => {
      promoteBtn.disabled = true;
      try {
        const res = await fetch("/api/manual/promote", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ class_name: currentClassName(), tile_id: c.tile_id, label_polygon: c.label_polygon }),
        });
        if (!res.ok) throw new Error(await res.text());
        const sample = await res.json();
        samples.push(sample);
        refreshSamplesLayer();
        renderSamplesList();
        promoteBtn.textContent = "Added";
      } catch (err) {
        promoteBtn.disabled = false;
        showWarning("Failed to add: " + err.message);
      }
    });
    wrap.appendChild(promoteBtn);
    validationResultsEl.appendChild(wrap);
  }
}

// ---------- classes ----------

async function loadClasses() {
  const res = await fetch("/api/classes");
  const { classes, parents } = await res.json();
  knownClassNames = new Set(classes);
  const current = classSelect.value;

  const topLevel = classes.filter((c) => !parents[c]);
  const childrenOf = (parent) => classes.filter((c) => parents[c] === parent);

  // Flat list, not <optgroup>: an optgroup's label is a bold but unselectable header, so keeping
  // the parent itself selectable meant *also* adding it as a plain option right below that
  // header -- which just showed "fence" twice. Indentation is done with non-breaking spaces
  // (plain spaces collapse when an <option> renders) instead of an arrow glyph, which doesn't
  // render in every font. A child's own name has its "<parent>/" prefix stripped since the
  // indentation under its parent already shows that relationship.
  classSelect.innerHTML = "";
  for (const top of topLevel) {
    const topOpt = document.createElement("option");
    topOpt.value = top;
    topOpt.textContent = top;
    classSelect.appendChild(topOpt);
    for (const kid of childrenOf(top)) {
      const kidOpt = document.createElement("option");
      kidOpt.value = kid;
      kidOpt.textContent = "    " + kid.slice(top.length + 1);
      classSelect.appendChild(kidOpt);
    }
  }
  if (classes.includes(current)) classSelect.value = current;

  const parentSelectValue = addClassParentSelect.value;
  addClassParentSelect.innerHTML = '<option value="">-- select parent --</option>';
  for (const top of topLevel) {
    const opt = document.createElement("option");
    opt.value = top;
    opt.textContent = top;
    addClassParentSelect.appendChild(opt);
  }
  if (topLevel.includes(parentSelectValue)) addClassParentSelect.value = parentSelectValue;

  updateAddClassCreateEnabled();

  // Whatever ends up selected -- restored above, or just the browser's default first option --
  // gets its samples loaded, regardless of whether the user actually interacted with the
  // dropdown. A <select>'s default selection doesn't fire a "change" event, so without this the
  // samples panel stayed empty until the user manually touched the dropdown.
  if (classSelect.value) await loadSamples();
}

classSelect.addEventListener("change", () => {
  loadSamples();
  if (trainingTab.style.display !== "none") loadTrainingPanel();
});
addClassToggleBtn.addEventListener("click", () => {
  if (addClassPanel.style.display === "none") openAddClassPanel();
  else closeAddClassPanel();
});
addClassTypeSelect.addEventListener("change", updateAddClassPanelState);
addClassParentSelect.addEventListener("change", updateAddClassCreateEnabled);
addClassNameInput.addEventListener("input", updateAddClassCreateEnabled);
addClassNameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !addClassCreateBtn.disabled) createNewClass();
});
addClassCreateBtn.addEventListener("click", createNewClass);

tabBtnSamples.addEventListener("click", () => switchTab("samples"));
tabBtnValidation.addEventListener("click", () => switchTab("validation"));
tabBtnTraining.addEventListener("click", () => switchTab("training"));
tabBtnGraph.addEventListener("click", () => switchTab("graph"));
generatePackageBtn.addEventListener("click", generatePackage);
openValidationModalBtn.addEventListener("click", openValidationModal);
validationPickPositionBtn.addEventListener("click", startPickingPosition);
validationModalCancel.addEventListener("click", () => { validationModal.style.display = "none"; });
validationModalRun.addEventListener("click", runValidationFromModal);
abortValidationBtn.addEventListener("click", abortValidation);
warningModalOk.addEventListener("click", () => { warningModal.style.display = "none"; });

loadConfig();
loadClasses();
