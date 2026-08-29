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
const classNewInput = document.getElementById("class-new-input");
const classNewParentSelect = document.getElementById("class-new-parent-select");

const tabBtnSamples = document.getElementById("tab-btn-samples");
const tabBtnValidation = document.getElementById("tab-btn-validation");
const tabBtnTraining = document.getElementById("tab-btn-training");
const samplesTab = document.getElementById("samples-tab");
const validationTab = document.getElementById("validation-tab");
const trainingTab = document.getElementById("training-tab");
const trainingTreeEl = document.getElementById("training-tree");
const trainingEpochsInput = document.getElementById("training-epochs-input");
const trainingPatienceInput = document.getElementById("training-patience-input");
const trainingBaseModelInput = document.getElementById("training-base-model-input");

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
  return classSelect.value === "__new__" ? classNewInput.value.trim() : classSelect.value;
}

function updateClassInputVisibility() {
  const isNew = classSelect.value === "__new__";
  classNewInput.style.display = isNew ? "block" : "none";
  classNewParentSelect.style.display = isNew ? "block" : "none";
  if (isNew) classNewInput.focus();
}

async function createNewClassAndLoad() {
  const name = classNewInput.value.trim();
  if (!name) return;
  if (knownClassNames.has(name)) {
    await loadSamples();
    return;
  }
  const parent = classNewParentSelect.value || null;
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
  classSelect.value = name;
  updateClassInputVisibility();
  await loadSamples();
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

  const check = await (await fetch("/api/validate_bbox", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...bbox, zoom }),
  })).json();
  if (!check.ok) {
    draw.delete(feature.id);
    showWarning(`Shape rejected: too small to use (~${check.min_px}px at zoom ${zoom}, need at least 150px). Redraw with more margin around the object.`);
    return;
  }

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

async function loadSamples() {
  const className = currentClassName();
  samples = [];
  editingSampleId = null;
  editingFeatureId = null;
  if (!className) {
    refreshSamplesLayer();
    renderSamplesList();
    return;
  }
  const res = await fetch(`/api/manual/samples?class_name=${encodeURIComponent(className)}`);
  const data = await res.json();
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
      ? (data.s3_key ? `Uploaded to S3 (${data.s3_key}).` : "S3 upload failed -- check logs.")
      : "S3 not configured -- kept local only.";
    const mergeNote = data.merge
      ? `Merged ${data.merge.added_from_remote} new sample(s) from S3 (had ${data.merge.local_total} local, latest S3 entry had ${data.merge.remote_total}). `
      : (includeLatestCheckbox.checked ? "No S3 package to merge yet -- packaged local samples only. " : "");
    generatePackageProgressEl.value = 100;
    generatePackageStatusEl.textContent =
      `${mergeNote}Done: seg ${data.segmentation.train}/${data.segmentation.val} (train/val), ` +
      `obb ${data.obb.train}/${data.obb.val} (train/val). ${s3Note} ` +
      `Train separately via scripts/train.py or scripts/train_obb.py -- this button doesn't train.`;
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
  tabBtnSamples.classList.toggle("active", tab === "samples");
  tabBtnValidation.classList.toggle("active", tab === "validation");
  tabBtnTraining.classList.toggle("active", tab === "training");
  if (tab === "training") loadTrainingTree();
}

// ---------- training tab ----------

async function loadTrainingTree() {
  const [classesRes, activeRes] = await Promise.all([
    fetch("/api/classes").then((r) => r.json()),
    fetch("/api/train/active").then((r) => r.json()),
  ]);
  const { classes, parents } = classesRes;
  const activeJobs = activeRes.jobs || {};

  const topLevel = classes.filter((c) => !parents[c]);
  const childrenOf = (parent) => classes.filter((c) => parents[c] === parent);

  trainingTreeEl.innerHTML = "";
  for (const top of topLevel) {
    trainingTreeEl.appendChild(buildTrainingRow(top, false));
    for (const kid of childrenOf(top)) {
      trainingTreeEl.appendChild(buildTrainingRow(kid, true));
    }
  }

  for (const [className, jobId] of Object.entries(activeJobs)) {
    const row = trainingTreeEl.querySelector(`[data-class="${CSS.escape(className)}"]`);
    if (row) watchTrainingJob(row, jobId);
  }
}

function buildTrainingRow(className, indented) {
  const row = document.createElement("div");
  row.className = "training-row" + (indented ? " indented" : "");
  row.dataset.class = className;

  const label = document.createElement("span");
  label.className = "training-row-label";
  label.textContent = (indented ? "↳ " : "") + className;
  row.appendChild(label);

  const status = document.createElement("span");
  status.className = "training-row-status";
  row.appendChild(status);

  const progress = document.createElement("progress");
  progress.max = 100;
  progress.value = 0;
  progress.style.display = "none";
  row.appendChild(progress);

  const btn = document.createElement("button");
  btn.className = "training-row-btn secondary";
  btn.textContent = "Train";
  btn.addEventListener("click", () => startTraining(className, row));
  row.appendChild(btn);

  return row;
}

async function startTraining(className, row) {
  const btn = row.querySelector(".training-row-btn");
  const status = row.querySelector(".training-row-status");
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
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    await watchTrainingJob(row, job_id);
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
  progress.style.display = "inline-block";

  const job = await pollJob(jobId, {
    intervalMs: 3000,
    onProgress: (p) => {
      progress.value = p.percent || 0;
      const metricsStr = p.metrics
        ? Object.entries(p.metrics).map(([k, v]) => `${k.split("/").pop()}=${v}`).join(" ")
        : "";
      status.textContent = `${p.step || "Working..."} ${metricsStr}`;
    },
  });

  progress.style.display = "none";
  btn.disabled = false;
  if (job.status === "error") {
    status.textContent = "Error: " + job.error;
  } else if (job.status === "done") {
    const m = job.result && job.result.metrics && job.result.metrics.metrics;
    const metricsStr = m
      ? Object.entries(m).map(([k, v]) => `${k.split("/").pop()}=${typeof v === "number" ? v.toFixed(3) : v}`).join(" ")
      : "";
    status.textContent = `Saved ${job.result.version}. ${metricsStr}`;
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

  classSelect.innerHTML = '<option value="__new__">+ New class</option>';
  for (const top of topLevel) {
    const kids = childrenOf(top);
    const group = document.createElement("optgroup");
    group.label = top;
    const topOpt = document.createElement("option");
    topOpt.value = top;
    topOpt.textContent = top;
    group.appendChild(topOpt);
    for (const kid of kids) {
      const kidOpt = document.createElement("option");
      kidOpt.value = kid;
      kidOpt.textContent = `↳ ${kid}`;
      group.appendChild(kidOpt);
    }
    classSelect.appendChild(group);
  }
  if (classes.includes(current)) classSelect.value = current;

  classNewParentSelect.innerHTML = '<option value="">(top-level class)</option>';
  for (const top of topLevel) {
    const opt = document.createElement("option");
    opt.value = top;
    opt.textContent = `sub-class of ${top}`;
    classNewParentSelect.appendChild(opt);
  }

  updateClassInputVisibility();
}

classSelect.addEventListener("change", () => {
  updateClassInputVisibility();
  loadSamples();
});
classNewInput.addEventListener("blur", createNewClassAndLoad);
classNewInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") createNewClassAndLoad();
});

tabBtnSamples.addEventListener("click", () => switchTab("samples"));
tabBtnValidation.addEventListener("click", () => switchTab("validation"));
tabBtnTraining.addEventListener("click", () => switchTab("training"));
generatePackageBtn.addEventListener("click", generatePackage);
openValidationModalBtn.addEventListener("click", openValidationModal);
validationPickPositionBtn.addEventListener("click", startPickingPosition);
validationModalCancel.addEventListener("click", () => { validationModal.style.display = "none"; });
validationModalRun.addEventListener("click", runValidationFromModal);
abortValidationBtn.addEventListener("click", abortValidation);
warningModalOk.addEventListener("click", () => { warningModal.style.display = "none"; });

loadConfig();
loadClasses();
