let map, draw;
let currentClass = null;
let currentRound = null;
let currentSeed = null; // {lat, lon} of the shape currently being processed
let currentJobId = null;
let candidates = []; // [{tile_id, similarity, thumbnail_url, rejected}]

let pendingShapes = []; // [{id, class_name, lat, lon}] drawn + labeled, not yet searched
let labelFeatures = []; // GeoJSON point features mirroring pendingShapes, for the label layer
let pendingDrawFeatureId = null; // feature awaiting a class name in the modal

const collectBtn = document.getElementById("collect-btn");
const abortBtn = document.getElementById("abort-btn");
const confirmBtn = document.getElementById("confirm-btn");
const statusEl = document.getElementById("status");
const seedInfoEl = document.getElementById("seed-info");
const resultsEl = document.getElementById("results");
const maxFetchesInput = document.getElementById("max-fetches-input");
const searchZoomInput = document.getElementById("search-zoom-input");
const thresholdInput = document.getElementById("threshold-input");

const classModal = document.getElementById("class-modal");
const classModalSelect = document.getElementById("class-modal-select");
const classModalInput = document.getElementById("class-modal-input");
const classModalOk = document.getElementById("class-modal-ok");
const classModalCancel = document.getElementById("class-modal-cancel");

const warningModal = document.getElementById("warning-modal");
const warningModalText = document.getElementById("warning-modal-text");
const warningModalOk = document.getElementById("warning-modal-ok");

function showWarning(text) {
  warningModalText.textContent = text;
  warningModal.style.display = "flex";
}

const tabBtnSearch = document.getElementById("tab-btn-search");
const tabBtnManage = document.getElementById("tab-btn-manage");
const searchTab = document.getElementById("search-tab");
const manageTab = document.getElementById("manage-tab");
const manageClassSelect = document.getElementById("manage-class-select");
const manageRoundsEl = document.getElementById("manage-rounds");
const packBtn = document.getElementById("pack-btn");
const packAbortBtn = document.getElementById("pack-abort-btn");
const packStatusEl = document.getElementById("pack-status");
const packEpochsInput = document.getElementById("pack-epochs-input");

function setStatus(text) {
  statusEl.textContent = text || "";
}

function updateFindExamplesEnabled() {
  collectBtn.disabled = pendingShapes.length === 0;
}

function polygonCentroid(feature) {
  const ring = feature.geometry.coordinates[0];
  const pts = ring.slice(0, -1); // last point repeats the first
  const lon = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const lat = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return { lon, lat };
}

function polygonBbox(feature) {
  const ring = feature.geometry.coordinates[0];
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  return {
    west: Math.min(...lons), east: Math.max(...lons),
    south: Math.min(...lats), north: Math.max(...lats),
  };
}

function refreshLabelLayer() {
  const source = map.getSource("shape-labels");
  if (source) source.setData({ type: "FeatureCollection", features: labelFeatures });
}

function updateClassModalInputVisibility() {
  const isNew = classModalSelect.value === "__new__";
  classModalInput.style.display = isNew ? "block" : "none";
  if (isNew) classModalInput.focus();
}

// A tightly-drawn shape (traced right up against a thin feature like a fence line, with no
// margin) can crop down to a handful of raw pixels -- too little for DINOv2 to embed usefully,
// as opposed to just being an unhelpfully-small search radius. Reject it up front rather than
// silently running a search that can never find anything (see fence_seed_4: a 31x98px crop
// matched nothing above 0.27 similarity out of 300 tiles checked).
async function validateAndOpenModal(feature) {
  const bbox = polygonBbox(feature);
  const res = await fetch("/api/validate_bbox", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...bbox, zoom: parseFloat(searchZoomInput.value) || 17 }),
  });
  const check = await res.json();
  if (!check.ok) {
    draw.delete(feature.id);
    showWarning(
      `Shape rejected: too small to search with (~${check.min_px}px at zoom ${searchZoomInput.value}, ` +
      `need at least 150px). Redraw with more margin around the object.`
    );
    return;
  }
  openClassModal(feature.id);
}

function openClassModal(featureId) {
  pendingDrawFeatureId = featureId;
  classModalSelect.value = "__new__";
  classModalInput.value = "";
  updateClassModalInputVisibility();
  classModal.style.display = "flex";
}

function closeClassModal() {
  classModal.style.display = "none";
  pendingDrawFeatureId = null;
}

function confirmClassModal() {
  const name = classModalSelect.value === "__new__" ? classModalInput.value.trim() : classModalSelect.value;
  if (!name) return;
  const feature = draw.get(pendingDrawFeatureId);
  const { lon, lat } = polygonCentroid(feature);
  const bbox = polygonBbox(feature);
  const polygon = feature.geometry.coordinates[0]; // [[lon,lat], ...] ring, incl. closing point

  pendingShapes.push({ id: pendingDrawFeatureId, class_name: name, lat, lon, polygon, ...bbox });
  labelFeatures.push({
    type: "Feature",
    geometry: { type: "Point", coordinates: [lon, lat] },
    properties: { class: name },
  });
  refreshLabelLayer();
  updateFindExamplesEnabled();
  closeClassModal();
}

function cancelClassModal() {
  if (pendingDrawFeatureId) draw.delete(pendingDrawFeatureId);
  closeClassModal();
}

async function loadConfig() {
  const res = await fetch("/api/config");
  const { mapbox_token } = await res.json();

  // Built from a raw raster source hitting the same Mapbox Raster Tiles API endpoint the
  // backend already uses (v4/mapbox.satellite/{z}/{x}/{y}), instead of a hosted Mapbox GL
  // style (mapbox://styles/...). The hosted style needs Styles/Fonts API access on top of
  // Tiles API access, which an enterprise token may not include; this only needs the Tiles
  // API, which we've already confirmed the token works for.
  const style = {
    version: 8,
    // Needed for the class-label symbol layer's text-field to render at all — without a glyphs
    // URL, map.addLayer() throws on any text-field layer (this is what caused the class label to
    // silently never appear). Free, no-token font CDN commonly used with MapLibre.
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
    map.addSource("shape-labels", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({
      id: "shape-labels-layer",
      type: "symbol",
      source: "shape-labels",
      layout: { "text-field": ["get", "class"], "text-font": ["Open Sans Regular"], "text-size": 14, "text-anchor": "center" },
      paint: { "text-color": "#ffffff", "text-halo-color": "#000000", "text-halo-width": 1.5 },
    });
  });

  // Fires once a polygon is completed (double-click) — this is the "validate the shape" moment.
  map.on("draw.create", (e) => validateAndOpenModal(e.features[0]));

  // mapbox-gl-draw defaults to a "grab" hand cursor even while actively placing vertices --
  // crosshair is the standard affordance for "you are drawing", so make that explicit instead
  // of relying on the library's default.
  map.on("draw.modechange", (e) => {
    map.getCanvas().style.cursor = e.mode === "draw_polygon" ? "crosshair" : "";
  });
}

async function loadClasses() {
  const res = await fetch("/api/classes");
  const { classes } = await res.json();
  const current = classModalSelect.value;
  classModalSelect.innerHTML = '<option value="__new__">+ New class</option>';
  for (const c of classes) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    classModalSelect.appendChild(opt);
  }
  if (classes.includes(current)) classModalSelect.value = current;
}

// A tile can legitimately contain more than one labeled instance -- each polygon renders as its
// own separate <polygon> element, never merged/concatenated into a single shape.
function labelOverlaySvg(labelPolygons) {
  if (!labelPolygons || !labelPolygons.length) return "";
  const shapes = labelPolygons
    .filter((poly) => poly && poly.length)
    .map((poly) => `<polygon points="${poly.map((p) => p.join(",")).join(" ")}" />`)
    .join("");
  return `<svg class="label-overlay" viewBox="0 0 1 1" preserveAspectRatio="none">${shapes}</svg>`;
}

function renderResults() {
  resultsEl.innerHTML = "";
  for (const c of candidates) {
    const div = document.createElement("div");
    div.className = "thumb" + (c.rejected ? " rejected" : "");
    const overlay = labelOverlaySvg(c.label_polygon);
    div.innerHTML = `<img src="${c.thumbnail_url}" />${overlay}<span class="sim">${c.similarity.toFixed(2)}</span>`;
    div.addEventListener("click", () => {
      c.rejected = !c.rejected;
      renderResults();
    });
    resultsEl.appendChild(div);
  }
  confirmBtn.style.display = candidates.length ? "block" : "none";
}

// Polls GET /api/jobs/{id} until it's no longer "running", calling onProgress along the way.
// Returns the final job object ({status, progress, result, error}).
async function pollJob(jobId, { onProgress, intervalMs = 1000 } = {}) {
  while (true) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (job.status !== "running") return job;
    if (onProgress) onProgress(job.progress);
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

async function collect() {
  if (!pendingShapes.length) return;
  const shape = pendingShapes.shift();
  currentClass = shape.class_name;
  currentSeed = { lat: shape.lat, lon: shape.lon, west: shape.west, south: shape.south, east: shape.east, north: shape.north, polygon: shape.polygon };
  updateFindExamplesEnabled();

  collectBtn.disabled = true;
  confirmBtn.style.display = "none";
  resultsEl.innerHTML = "";
  seedInfoEl.textContent = "";
  abortBtn.style.display = "block";
  setStatus(`Starting search for "${currentClass}"...`);

  try {
    const res = await fetch("/api/collect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        class_name: currentClass,
        lat: currentSeed.lat,
        lon: currentSeed.lon,
        // Independent of the map's current/draw-time zoom -- this is the zoom the seed crop and
        // every candidate tile actually get fetched/rendered at (see search-zoom-input's hint).
        zoom: parseFloat(searchZoomInput.value) || 17,
        west: currentSeed.west,
        south: currentSeed.south,
        east: currentSeed.east,
        north: currentSeed.north,
        polygon: currentSeed.polygon,
        threshold: parseFloat(thresholdInput.value) || 0.75,
        max_fetches: parseInt(maxFetchesInput.value, 10) || 300,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    currentJobId = job_id;

    const job = await pollJob(job_id, {
      onProgress: (p) => setStatus(
        `Searching "${currentClass}"... ${p.fetched_count || 0} tile(s) checked, ${p.candidates_found || 0} found so far.`
      ),
    });
    currentJobId = null;
    abortBtn.style.display = "none";

    if (job.status === "error") throw new Error(job.error);

    const data = job.result;
    currentRound = data.round;
    candidates = data.candidates.map((c) => ({ ...c, rejected: false }));

    seedInfoEl.textContent =
      `"${currentClass}" · round ${data.round} · ~${data.seed.meters_per_pixel.toFixed(2)} m/pixel · ` +
      `${data.exemplar_count} exemplar(s) used · fetched ${data.fetched_count} tile(s)` +
      (data.seed_added_to_dataset ? " · seed shape saved to dataset directly" : "");

    if (job.status === "aborted") {
      setStatus(`Aborted: ${candidates.length} candidate(s) found before stopping.`);
    } else if (data.stopped_reason === "max_fetches") {
      setStatus(`Stopped early: hit the search budget with only ${candidates.length} match(es). Try raising the budget or a different seed.`);
    } else if (!candidates.length) {
      setStatus("No matching candidates found near this seed.");
    } else {
      setStatus(`${candidates.length} candidates found. Click any thumbnail to toggle reject, then Confirm Round.`);
    }
    renderResults();
  } catch (err) {
    setStatus("Error: " + err.message);
    abortBtn.style.display = "none";
  } finally {
    updateFindExamplesEnabled();
  }
}

async function abortSearch() {
  if (!currentJobId) return;
  abortBtn.disabled = true;
  try {
    await fetch(`/api/jobs/${currentJobId}/abort`, { method: "POST" });
  } finally {
    abortBtn.disabled = false;
  }
}

async function confirmRound() {
  confirmBtn.disabled = true;
  const kept = candidates.filter((c) => !c.rejected).map((c) => c.tile_id);
  try {
    const res = await fetch("/api/reconcile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_name: currentClass, round: currentRound, kept_tile_ids: kept }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    setStatus(`Round ${currentRound} confirmed: ${data.confirmed_count} kept, ${data.rejected_count} rejected.`);
    candidates = [];
    resultsEl.innerHTML = "";
    confirmBtn.style.display = "none";
    loadClasses(); // in case this was a brand-new class
  } catch (err) {
    setStatus("Error: " + err.message);
  } finally {
    confirmBtn.disabled = false;
  }
}

// ---------- tabs ----------

function switchTab(tab) {
  const isSearch = tab === "search";
  searchTab.style.display = isSearch ? "block" : "none";
  manageTab.style.display = isSearch ? "none" : "block";
  tabBtnSearch.classList.toggle("active", isSearch);
  tabBtnManage.classList.toggle("active", !isSearch);
  if (!isSearch) loadManageClasses();
}

// ---------- manage tab ----------

async function loadManageClasses() {
  const res = await fetch("/api/classes");
  const { classes } = await res.json();
  const current = manageClassSelect.value;
  manageClassSelect.innerHTML = '<option value="">Select a class...</option>';
  for (const c of classes) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    manageClassSelect.appendChild(opt);
  }
  if (classes.includes(current)) {
    manageClassSelect.value = current;
    loadRounds();
  }
}

async function loadRounds() {
  const className = manageClassSelect.value;
  manageRoundsEl.innerHTML = "";
  if (!className) return;

  const res = await fetch(`/api/classes/${encodeURIComponent(className)}/rounds`);
  const { rounds } = await res.json();
  if (!rounds.length) {
    manageRoundsEl.innerHTML = '<p class="hint">No rounds yet for this class.</p>';
    return;
  }

  for (const r of rounds) {
    const section = document.createElement("div");
    section.className = "round-section";

    const header = document.createElement("div");
    header.className = "round-header";
    const label = document.createElement("span");
    label.textContent = `Round ${r.round}${r.seed_tile_id ? " (has seed)" : ""} · ${r.confirmed.length} kept · ` +
      `${r.pending.length} pending review · ${r.rejected_count} rejected`;
    const delRoundBtn = document.createElement("button");
    delRoundBtn.textContent = "Delete Round";
    delRoundBtn.addEventListener("click", async () => {
      if (!confirm(`Discard all ${r.pending.length} pending candidate(s) from round ${r.round}? ` +
        `(${r.confirmed.length} already-confirmed example(s) will be left untouched.)`)) return;
      await fetch(`/api/classes/${encodeURIComponent(className)}/rounds/${r.round}`, { method: "DELETE" });
      loadRounds();
    });
    header.appendChild(label);
    header.appendChild(delRoundBtn);
    section.appendChild(header);

    const thumbs = document.createElement("div");
    thumbs.className = "round-thumbs";
    for (const tid of r.confirmed) {
      const t = document.createElement("div");
      t.className = "round-thumb";
      const img = document.createElement("img");
      img.src = `/api/classes/${encodeURIComponent(className)}/dataset_image/${tid}`;
      const delBtn = document.createElement("button");
      delBtn.textContent = "✕";
      delBtn.title = "Delete this example";
      delBtn.addEventListener("click", async () => {
        await fetch(`/api/classes/${encodeURIComponent(className)}/examples/${tid}`, { method: "DELETE" });
        loadRounds();
      });
      t.appendChild(img);
      t.appendChild(delBtn);
      thumbs.appendChild(t);
    }
    section.appendChild(thumbs);

    // Candidates already sitting in the registry as pending_review -- e.g. from a search that
    // was aborted, or the browser tab was closed/refreshed before Confirm Round ran. Same
    // keep/reject-then-confirm flow as the live Search tab, just reading from disk instead of
    // in-memory JS state, and reusing the same /api/reconcile endpoint.
    if (r.pending.length) {
      const pendingWrap = document.createElement("div");
      pendingWrap.className = "pending-section";
      const pendingLabel = document.createElement("p");
      pendingLabel.className = "hint";
      pendingLabel.textContent = "Pending review — click a thumbnail to toggle reject, then Confirm.";
      pendingWrap.appendChild(pendingLabel);

      const pendingThumbs = document.createElement("div");
      pendingThumbs.className = "round-thumbs";
      const rejectedSet = new Set();
      for (const p of r.pending) {
        const tid = p.tile_id;
        const t = document.createElement("div");
        t.className = "round-thumb pending-thumb";
        t.innerHTML = `<img src="/api/tile_image/${encodeURIComponent(className)}/${tid}" />${labelOverlaySvg(p.label_polygon)}`;
        t.addEventListener("click", () => {
          if (rejectedSet.has(tid)) rejectedSet.delete(tid);
          else rejectedSet.add(tid);
          t.classList.toggle("rejected", rejectedSet.has(tid));
        });
        pendingThumbs.appendChild(t);
      }
      pendingWrap.appendChild(pendingThumbs);

      const confirmPendingBtn = document.createElement("button");
      confirmPendingBtn.textContent = "Confirm Pending";
      confirmPendingBtn.addEventListener("click", async () => {
        confirmPendingBtn.disabled = true;
        const kept = r.pending.map((p) => p.tile_id).filter((tid) => !rejectedSet.has(tid));
        try {
          await fetch("/api/reconcile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ class_name: className, round: r.round, kept_tile_ids: kept }),
          });
          loadRounds();
        } finally {
          confirmPendingBtn.disabled = false;
        }
      });
      pendingWrap.appendChild(confirmPendingBtn);
      section.appendChild(pendingWrap);
    }

    manageRoundsEl.appendChild(section);
  }
}

let packJobId = null;

async function packData() {
  packBtn.disabled = true;
  packAbortBtn.style.display = "block";
  packStatusEl.textContent = "Starting...";
  try {
    const res = await fetch("/api/pack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: parseInt(packEpochsInput.value, 10) || 20 }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    packJobId = job_id;

    const job = await pollJob(job_id, {
      intervalMs: 2000,
      onProgress: (p) => {
        packStatusEl.textContent = p.current_class
          ? `Training ${p.current_class} (${p.class_index}/${p.total_classes})...`
          : "Starting...";
      },
    });
    packJobId = null;
    packAbortBtn.style.display = "none";

    if (job.status === "error") throw new Error(job.error);
    const trained = (job.result && job.result.trained) || [];
    packStatusEl.textContent = job.status === "aborted"
      ? `Aborted. Trained ${trained.length} class(es) before stopping.`
      : `Done. Trained: ${trained.map((t) => `${t.class} ${t.version}`).join(", ") || "(none — no classes had data)"}`;
  } catch (err) {
    packStatusEl.textContent = "Error: " + err.message;
    packAbortBtn.style.display = "none";
  } finally {
    packBtn.disabled = false;
  }
}

async function abortPack() {
  if (!packJobId) return;
  packAbortBtn.disabled = true;
  try {
    await fetch(`/api/jobs/${packJobId}/abort`, { method: "POST" });
  } finally {
    packAbortBtn.disabled = false;
  }
}

collectBtn.addEventListener("click", collect);
abortBtn.addEventListener("click", abortSearch);
confirmBtn.addEventListener("click", confirmRound);
warningModalOk.addEventListener("click", () => { warningModal.style.display = "none"; });
classModalOk.addEventListener("click", confirmClassModal);
classModalCancel.addEventListener("click", cancelClassModal);
classModalSelect.addEventListener("change", updateClassModalInputVisibility);
classModalInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") confirmClassModal();
  if (e.key === "Escape") cancelClassModal();
});
tabBtnSearch.addEventListener("click", () => switchTab("search"));
tabBtnManage.addEventListener("click", () => switchTab("manage"));
manageClassSelect.addEventListener("change", loadRounds);
packBtn.addEventListener("click", packData);
packAbortBtn.addEventListener("click", abortPack);

loadConfig();
loadClasses();
