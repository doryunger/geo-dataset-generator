let map, draw;
let currentJobId = null;
let samples = []; // [{id, class_name, lon, lat, polygon, thumbnail_url}]
let editingSampleId = null; // sample currently pulled into `draw` for editing, or null
let editingFeatureId = null; // the id mapbox-gl-draw assigned that feature
let pickingValidationOrigin = false;
let pickedOrigin = null; // {lon, lat} chosen via "Pick on Map", or null if not set yet
let validationCandidates = []; // last validation run's results

const classSelect = document.getElementById("class-select");
const classNewInput = document.getElementById("class-new-input");

const tabBtnSamples = document.getElementById("tab-btn-samples");
const tabBtnValidation = document.getElementById("tab-btn-validation");
const samplesTab = document.getElementById("samples-tab");
const validationTab = document.getElementById("validation-tab");

const samplesListEl = document.getElementById("samples-list");
const generatePackageBtn = document.getElementById("generate-package-btn");
const generatePackageStatusEl = document.getElementById("generate-package-status");

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

function currentClassName() {
  return classSelect.value === "__new__" ? classNewInput.value.trim() : classSelect.value;
}

function updateClassInputVisibility() {
  const isNew = classSelect.value === "__new__";
  classNewInput.style.display = isNew ? "block" : "none";
  if (isNew) classNewInput.focus();
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
  const res = await fetch(`/api/manual/samples/${encodeURIComponent(currentClassName())}/${sampleId}`, {
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
      await fetch(`/api/manual/samples/${encodeURIComponent(currentClassName())}/${s.id}`, { method: "DELETE" });
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
  generatePackageStatusEl.textContent = "Generating...";
  try {
    const res = await fetch("/api/manual/generate_package", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_name: className }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    generatePackageStatusEl.textContent = `Done: ${data.train} train, ${data.val} val. Use Pack Data (main app's Manage tab) to train.`;
  } catch (err) {
    generatePackageStatusEl.textContent = "Error: " + err.message;
  } finally {
    generatePackageBtn.disabled = false;
  }
}

// ---------- validation tab ----------

function switchTab(tab) {
  const isSamples = tab === "samples";
  samplesTab.style.display = isSamples ? "block" : "none";
  validationTab.style.display = isSamples ? "none" : "block";
  tabBtnSamples.classList.toggle("active", isSamples);
  tabBtnValidation.classList.toggle("active", !isSamples);
}

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
      setValidationStatus(`${validationCandidates.length} candidate(s) found from ${job.result.exemplar_count} sample(s), ${job.result.fetched_count} tile(s) checked.`);
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
  const { classes } = await res.json();
  const current = classSelect.value;
  classSelect.innerHTML = '<option value="__new__">+ New class</option>';
  for (const c of classes) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    classSelect.appendChild(opt);
  }
  if (classes.includes(current)) classSelect.value = current;
}

classSelect.addEventListener("change", () => {
  updateClassInputVisibility();
  loadSamples();
});
classNewInput.addEventListener("blur", loadSamples);
classNewInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadSamples();
});

tabBtnSamples.addEventListener("click", () => switchTab("samples"));
tabBtnValidation.addEventListener("click", () => switchTab("validation"));
generatePackageBtn.addEventListener("click", generatePackage);
openValidationModalBtn.addEventListener("click", openValidationModal);
validationPickPositionBtn.addEventListener("click", startPickingPosition);
validationModalCancel.addEventListener("click", () => { validationModal.style.display = "none"; });
validationModalRun.addEventListener("click", runValidationFromModal);
abortValidationBtn.addEventListener("click", abortValidation);
warningModalOk.addEventListener("click", () => { warningModal.style.display = "none"; });

loadConfig();
loadClasses();
