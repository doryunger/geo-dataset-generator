let map, draw, drawControlsEl;
let currentJobId = null;
let samples = [];
let editingSampleId = null;
let editingFeatureId = null;
let knownClassNames = new Set();
let addingHardNegative = false;

const classSelect = document.getElementById("class-select");
const addClassToggleBtn = document.getElementById("add-class-toggle-btn");
const addClassPanel = document.getElementById("add-class-panel");
const addClassTypeSelect = document.getElementById("add-class-type-select");
const addClassParentLabel = document.getElementById("add-class-parent-label");
const addClassParentSelect = document.getElementById("add-class-parent-select");
const addClassNameInput = document.getElementById("add-class-name-input");
const addClassCreateBtn = document.getElementById("add-class-create-btn");

const tabBtnSamples = document.getElementById("tab-btn-samples");
const tabBtnTraining = document.getElementById("tab-btn-training");
const samplesTab = document.getElementById("samples-tab");
const trainingTab = document.getElementById("training-tab");
const trainingTreeEl = document.getElementById("training-tree");
const trainingEpochsInput = document.getElementById("training-epochs-input");
const trainingPatienceInput = document.getElementById("training-patience-input");
const trainingBaseModelInput = document.getElementById("training-base-model-input");

const tabBtnHardNegatives = document.getElementById("tab-btn-hard-negatives");
const hardNegativesTab = document.getElementById("hard-negatives-tab");
const addHardNegativeBtn = document.getElementById("add-hard-negative-btn");
const hardNegativesStatusEl = document.getElementById("hard-negatives-status");
const hardNegativesListEl = document.getElementById("hard-negatives-list");
const includeHardNegativesCheckbox = document.getElementById("include-hard-negatives-checkbox");

const samplesListEl = document.getElementById("samples-list");
const generatePackageBtn = document.getElementById("generate-package-btn");
const generatePackageProgressEl = document.getElementById("generate-package-progress");
const generatePackageStatusEl = document.getElementById("generate-package-status");
const includeLatestCheckbox = document.getElementById("include-latest-checkbox");

const warningModal = document.getElementById("warning-modal");
const warningModalText = document.getElementById("warning-modal-text");
const warningModalOk = document.getElementById("warning-modal-ok");

const uploadLayerModal = document.getElementById("upload-layer-modal");
const uploadLayerFilePath = document.getElementById("upload-layer-file-path");
const uploadLayerBrowseBtn = document.getElementById("upload-layer-browse-btn");
const uploadLayerFileInput = document.getElementById("upload-layer-file-input");
const uploadLayerStatusEl = document.getElementById("upload-layer-status");
const uploadLayerCloseBtn = document.getElementById("upload-layer-close-btn");

function showWarning(text) {
  warningModalText.textContent = text;
  warningModal.style.display = "flex";
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && addingHardNegative) stopAddingHardNegative();
});

function currentClassName() {
  return classSelect.value;
}

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

function polygonCentroid(ring) {
  const pts = ring.slice(0, -1);
  const lon = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const lat = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return { lon, lat };
}

function polygonBbox(ring) {
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  return { west: Math.min(...lons), east: Math.max(...lons), south: Math.min(...lats), north: Math.max(...lats) };
}

function showPanelSpinner(container) {
  container.innerHTML = '<div class="panel-spinner"></div>';
}

function beginRowAction(listEl, btn) {
  listEl.querySelectorAll("button").forEach((b) => { b.hidden = true; });
  const spinner = document.createElement("div");
  spinner.className = "row-spinner";
  btn.replaceWith(spinner);
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
  drawControlsEl = document.querySelector(".mapboxgl-ctrl-group");
  updateDrawControlsVisibility();

  map.addControl({
    onAdd() {
      this._container = document.createElement("div");
      this._container.className = "maplibregl-ctrl maplibregl-ctrl-group";
      const btn = document.createElement("button");
      btn.className = "upload-layer-ctrl-btn";
      btn.type = "button";
      btn.textContent = "Upload Data Layer";
      btn.style.cssText =
        "width:auto; height:auto; margin:0; padding:8px 12px; white-space:nowrap; " +
        "background:#1a73e8; color:#fff; border:none; border-radius:4px; " +
        "font-size:13px; font-weight:600; cursor:pointer;";
      btn.addEventListener("click", openUploadLayerModal);
      this._container.appendChild(btn);
      return this._container;
    },
    onRemove() {
      this._container.remove();
    },
  });

  map.on("load", () => {
    map.addSource("samples-source", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({
      id: "samples-fill", type: "fill", source: "samples-source",
      paint: { "fill-color": "#3b82f6", "fill-opacity": 0.2 },
    });
    map.addLayer({
      id: "samples-line", type: "line", source: "samples-source",
      paint: { "line-color": "#3b82f6", "line-width": 2 },
    });

    map.addSource("vertex-dots-source", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({
      id: "vertex-dots", type: "circle", source: "vertex-dots-source",
      paint: { "circle-radius": 4, "circle-color": "#3b82f6", "circle-stroke-width": 1, "circle-stroke-color": "#fff" },
    });

    map.addSource("hard-negatives-source", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({
      id: "hard-negatives-fill", type: "fill", source: "hard-negatives-source",
      paint: {
        "fill-color": ["case", ["get", "enabled"], "#e74c3c", "#999"],
        "fill-opacity": ["case", ["get", "enabled"], 0.15, 0.08],
      },
    });
    map.addLayer({
      id: "hard-negatives-line", type: "line", source: "hard-negatives-source",
      paint: {
        "line-color": ["case", ["get", "enabled"], "#e74c3c", "#999"],
        "line-width": 2,
        "line-dasharray": ["case", ["get", "enabled"], ["literal", [1, 0]], ["literal", [2, 2]]],
      },
    });

    refreshSamplesLayer();
  });

  map.on("draw.create", (e) => {
    if (addingHardNegative) handleNewHardNegativeShape(e.features[0]);
    else handleNewShape(e.features[0]);
  });

  map.on("draw.modechange", (e) => {
    map.getCanvas().style.cursor = e.mode === "draw_polygon" ? "crosshair" : "";
    if (e.mode !== "draw_polygon") clearVertexDots();
  });
  map.on("draw.render", () => {
    if (draw.getMode() === "draw_polygon") updateVertexDots();
  });

  map.on("dblclick", "samples-fill", (e) => {
    e.preventDefault();
    const sampleId = e.features[0].properties.sampleId;
    startEditingSample(sampleId);
  });

  map.on("draw.selectionchange", (e) => {
    if (editingSampleId && e.features.length === 0) finishEditingSample();
  });
}

function clearVertexDots() {
  const source = map.getSource("vertex-dots-source");
  if (source) source.setData({ type: "FeatureCollection", features: [] });
}

function updateVertexDots() {
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

const GEOJSON_TYPES = new Set([
  "FeatureCollection", "Feature", "Point", "MultiPoint", "LineString",
  "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection",
]);

function openUploadLayerModal() {
  uploadLayerFilePath.value = "";
  uploadLayerFileInput.value = "";
  uploadLayerStatusEl.textContent = "";
  uploadLayerModal.style.display = "flex";
}

function applyUploadedLayer(geojson) {
  const source = map.getSource("uploaded-data");
  if (source) {
    source.setData(geojson);
  } else {
    map.addSource("uploaded-data", { type: "geojson", data: geojson });
  }
  if (!map.getLayer("uploaded-data-layer")) {
    map.addLayer({
      id: "uploaded-data-layer",
      type: "line",
      source: "uploaded-data",
      minzoom: 3,
      paint: { "line-color": "#e74c3c", "line-width": 3 },
    });
  }
}

function handleUploadLayerFile(file) {
  if (!file) return;
  uploadLayerFilePath.value = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    let geojson;
    try {
      geojson = JSON.parse(reader.result);
    } catch (err) {
      uploadLayerStatusEl.textContent = "Invalid file: not valid JSON.";
      return;
    }
    if (!geojson || typeof geojson !== "object" || !GEOJSON_TYPES.has(geojson.type)) {
      uploadLayerStatusEl.textContent = "Invalid file: not a GeoJSON object.";
      return;
    }
    applyUploadedLayer(geojson);
    uploadLayerStatusEl.textContent = "Layer loaded.";
  };
  reader.onerror = () => {
    uploadLayerStatusEl.textContent = "Failed to read file.";
  };
  reader.readAsText(file);
}

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
  draw.delete(feature.id);

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

function startEditingSample(sampleId) {
  if (editingSampleId) return;
  const sample = samples.find((s) => s.id === sampleId);
  if (!sample) return;
  editingSampleId = sampleId;
  const [id] = draw.add({ type: "Feature", geometry: { type: "Polygon", coordinates: [sample.polygon] }, properties: {} });
  editingFeatureId = id;
  refreshSamplesLayer();
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

let samplesRequestId = 0;

async function loadSamples() {
  const className = currentClassName();
  const requestId = ++samplesRequestId;

  samples = [];
  editingSampleId = null;
  editingFeatureId = null;
  refreshSamplesLayer();
  if (!className) {
    renderSamplesList();
    return;
  }
  showPanelSpinner(samplesListEl);

  const res = await fetch(`/api/manual/samples?class_name=${encodeURIComponent(className)}`);
  const data = await res.json();
  if (requestId !== samplesRequestId) return;
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
      beginRowAction(samplesListEl, delBtn);
      try {
        await fetch(`/api/manual/samples/${s.id}?class_name=${encodeURIComponent(currentClassName())}`, { method: "DELETE" });
        samples = samples.filter((x) => x.id !== s.id);
        refreshSamplesLayer();
      } catch (err) {
        showWarning("Failed to delete: " + err.message);
      }
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
      body: JSON.stringify({
        class_name: className,
        include_latest: includeLatestCheckbox.checked,
        include_hard_negatives: includeHardNegativesCheckbox.checked,
      }),
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

function updateDrawControlsVisibility() {
  if (!drawControlsEl) return;
  const visible = samplesTab.style.display !== "none" || addingHardNegative;
  drawControlsEl.style.display = visible ? "" : "none";
}

function switchTab(tab) {
  if (addingHardNegative && tab !== "hard-negatives") stopAddingHardNegative();
  samplesTab.style.display = tab === "samples" ? "block" : "none";
  trainingTab.style.display = tab === "training" ? "block" : "none";
  hardNegativesTab.style.display = tab === "hard-negatives" ? "block" : "none";
  tabBtnSamples.classList.toggle("active", tab === "samples");
  tabBtnTraining.classList.toggle("active", tab === "training");
  tabBtnHardNegatives.classList.toggle("active", tab === "hard-negatives");
  if (tab === "training") loadTrainingPanel();
  if (tab === "hard-negatives") loadHardNegatives(true);
  updateDrawControlsVisibility();
}

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
  progress.style.visibility = "hidden";
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

function startAddingHardNegative() {
  if (!currentClassName()) {
    showWarning("Pick or name a class first.");
    return;
  }
  addingHardNegative = true;
  addHardNegativeBtn.textContent = "Draw a shape... (Esc to cancel)";
  hardNegativesStatusEl.textContent = "";
  updateDrawControlsVisibility();
  draw.changeMode("draw_polygon");
}

function stopAddingHardNegative() {
  addingHardNegative = false;
  addHardNegativeBtn.textContent = "+ Add Hard Negative";
  updateDrawControlsVisibility();
  if (draw.getMode() === "draw_polygon") draw.changeMode("simple_select");
}

async function handleNewHardNegativeShape(feature) {
  const className = currentClassName();
  const ring = feature.geometry.coordinates[0];
  draw.delete(feature.id);
  stopAddingHardNegative();

  hardNegativesStatusEl.textContent = "Adding...";
  try {
    const res = await fetch("/api/manual/hard_negatives", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_name: className, polygon: ring }),
    });
    if (!res.ok) throw new Error(await res.text());
    hardNegativesStatusEl.textContent = "";
    await loadHardNegatives();
  } catch (err) {
    hardNegativesStatusEl.textContent = "Error: " + err.message;
  }
}

let hardNegativesRequestId = 0;

async function loadHardNegatives(clearFirst = false) {
  const className = currentClassName();
  const requestId = ++hardNegativesRequestId;
  if (!className) {
    hardNegativesListEl.innerHTML = "";
    refreshHardNegativesLayer([]);
    return;
  }
  if (clearFirst) {
    showPanelSpinner(hardNegativesListEl);
    refreshHardNegativesLayer([]);
  }
  const res = await fetch(`/api/manual/hard_negatives?class_name=${encodeURIComponent(className)}`);
  const data = await res.json();
  if (requestId !== hardNegativesRequestId) return;
  renderHardNegativesList(data.tiles || []);
  refreshHardNegativesLayer(data.tiles || []);
}

let hardNegativeTiles = [];

function refreshHardNegativesLayer(tiles) {
  const source = map.getSource("hard-negatives-source");
  if (!source) return;
  const features = tiles.map((t) => ({
    type: "Feature", properties: { id: t.id, enabled: t.enabled }, geometry: { type: "Polygon", coordinates: [t.polygon] },
  }));
  source.setData({ type: "FeatureCollection", features });
}

function renderHardNegativesList(tiles) {
  hardNegativeTiles = tiles;
  hardNegativesListEl.innerHTML = "";
  for (const t of tiles) {
    const row = document.createElement("div");
    row.className = "sample-row";
    row.classList.toggle("hard-negative-disabled", !t.enabled);
    const img = document.createElement("img");
    img.src = t.thumbnail_url;
    row.appendChild(img);

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "hard-negative-toggle";
    toggleLabel.title = "Include in training";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = t.enabled;
    toggle.addEventListener("click", (e) => e.stopPropagation());
    toggle.addEventListener("change", async () => {
      toggle.disabled = true;
      try {
        const res = await fetch(`/api/manual/hard_negatives/${t.id}?class_name=${encodeURIComponent(currentClassName())}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: toggle.checked }),
        });
        if (!res.ok) throw new Error(await res.text());
        t.enabled = toggle.checked;
        row.classList.toggle("hard-negative-disabled", !t.enabled);
        refreshHardNegativesLayer(hardNegativeTiles);
      } catch (err) {
        toggle.checked = !toggle.checked;
        hardNegativesStatusEl.textContent = "Error: " + err.message;
      }
      toggle.disabled = false;
    });
    toggleLabel.appendChild(toggle);
    row.appendChild(toggleLabel);

    const delBtn = document.createElement("button");
    delBtn.className = "danger";
    delBtn.textContent = "✕";
    delBtn.title = "Remove this hard negative";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      beginRowAction(hardNegativesListEl, delBtn);
      try {
        await fetch(`/api/manual/hard_negatives/${t.id}?class_name=${encodeURIComponent(currentClassName())}`, { method: "DELETE" });
      } catch (err) {
        hardNegativesStatusEl.textContent = "Error: " + err.message;
      }
      await loadHardNegatives();
    });
    row.appendChild(delBtn);
    row.addEventListener("click", () => {
      const bbox = polygonBbox(t.polygon);
      map.fitBounds([[bbox.west, bbox.south], [bbox.east, bbox.north]], { padding: 60 });
    });
    hardNegativesListEl.appendChild(row);
  }
}

async function loadClasses() {
  const res = await fetch("/api/classes");
  const { classes, parents } = await res.json();
  knownClassNames = new Set(classes);
  const current = classSelect.value;

  const topLevel = classes.filter((c) => !parents[c]);
  const childrenOf = (parent) => classes.filter((c) => parents[c] === parent);

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

  if (classSelect.value) await loadSamples();
}

classSelect.addEventListener("change", () => {
  loadSamples();
  if (trainingTab.style.display !== "none") loadTrainingPanel();
  if (hardNegativesTab.style.display !== "none") loadHardNegatives(true);
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
tabBtnTraining.addEventListener("click", () => switchTab("training"));
tabBtnHardNegatives.addEventListener("click", () => switchTab("hard-negatives"));
addHardNegativeBtn.addEventListener("click", () => {
  if (addingHardNegative) stopAddingHardNegative();
  else startAddingHardNegative();
});
generatePackageBtn.addEventListener("click", generatePackage);
warningModalOk.addEventListener("click", () => { warningModal.style.display = "none"; });
uploadLayerBrowseBtn.addEventListener("click", () => uploadLayerFileInput.click());
uploadLayerFileInput.addEventListener("change", () => handleUploadLayerFile(uploadLayerFileInput.files[0]));
uploadLayerCloseBtn.addEventListener("click", () => { uploadLayerModal.style.display = "none"; });

loadConfig();
loadClasses();
