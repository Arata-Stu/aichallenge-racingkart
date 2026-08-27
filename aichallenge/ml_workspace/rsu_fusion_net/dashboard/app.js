const SENSOR_COLORS = ["#c9fa3f", "#ffb347", "#ff6b79", "#b58cff", "#55b8ff", "#ff79cf", "#74e66c"];
const state = {
  overview: null, detail: null, frame: null, prediction: null,
  activeVersion: new URLSearchParams(location.search).get("version") || localStorage.getItem("rsuFusionVersion") || "default",
  sensor: 0, view: "sensor", playing: false, timer: null, selectedJob: null,
  selectedEvaluation: null, course: null, courseFrame: null,
  coursePlaying: false, courseTimer: null, courseRequest: 0, courseTransform: null,
  collectionSelected: null, collectionLoaded: null, collectionStamp: null,
  collectionLatest: null, collectionPolling: false,
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[character]);
const formatNumber = (value) => new Intl.NumberFormat("ja-JP").format(value || 0);
const formatTime = (value) => value ? new Date(value * 1000).toLocaleString("ja-JP") : "—";
function formatBytes(value) {
  if (!value) return "0 B";
  const index = Math.min(3, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${["B", "KB", "MB", "GB"][index]}`;
}
function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast show${error ? " error" : ""}`;
  clearTimeout(element._timer);
  element._timer = setTimeout(() => { element.className = "toast"; }, 3500);
}

async function refreshOverview() {
  const previous = $("#sequenceSelect").value;
  try {
    state.overview = await api("/api/overview");
    $("#serverState").textContent = `ONLINE · PID ${state.overview.server.pid}`;
    $(".server-state").classList.add("online");
    renderVersions(); renderMetrics(); renderSequences(previous);
    renderRecordings(); renderCollection(); renderTraining(); renderEvaluations(); renderJobs();
  } catch (error) {
    $("#serverState").textContent = "OFFLINE";
    $(".server-state").classList.remove("online");
    toast(error.message, true);
  }
}

function versionSequences(split = "") {
  return state.overview.sequences.filter((item) => item.version === state.activeVersion && (!split || item.split === split));
}
function renderVersions() {
  const versions = state.overview.versions;
  const items = versions.some((item) => item.id === state.activeVersion) ? versions : [...versions, { id: state.activeVersion, pending: true }];
  const options = items.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)}${item.pending ? " · 作成中" : ""}</option>`).join("");
  $("#datasetVersion").innerHTML = options; $("#preprocessVersion").innerHTML = options;
  $("#datasetVersion").value = state.activeVersion; $("#preprocessVersion").value = state.activeVersion;
  $("#activeVersionLabel").textContent = state.activeVersion; $("#trainingVersionLabel").textContent = state.activeVersion; $("#evaluationVersionLabel").textContent = state.activeVersion;
  const metadata = versions.find((item) => item.id === state.activeVersion);
  $("#versionSummary").textContent = metadata ? `Train ${formatNumber(metadata.train_samples)} · Val ${formatNumber(metadata.val_samples)}` : "前処理ジョブが作成中です";
}
function renderMetrics() {
  const train = versionSequences("train"), validation = versionSequences("val");
  $("#recordingCount").textContent = formatNumber(state.overview.recordings.length);
  $("#trainSamples").textContent = formatNumber(train.reduce((sum, item) => sum + item.samples, 0));
  $("#valSamples").textContent = formatNumber(validation.reduce((sum, item) => sum + item.samples, 0));
  $("#trainSequences").textContent = `${train.length} sequences`; $("#valSequences").textContent = `${validation.length} sequences`;
  $("#checkpointCount").textContent = formatNumber(state.overview.checkpoints.length);
  $("#runningBadge").hidden = !state.overview.jobs.some((job) => ["queued", "running"].includes(job.status));
}
function selectedSequence() {
  const [split = "", id = ""] = $("#sequenceSelect").value.split("|");
  return { version: state.activeVersion, split: decodeURIComponent(split), id: decodeURIComponent(id) };
}
function renderSequences(previous = "") {
  const sequences = versionSequences();
  const select = $("#sequenceSelect");
  select.innerHTML = sequences.map((item) => `<option value="${encodeURIComponent(item.split)}|${encodeURIComponent(item.id)}">${item.split.toUpperCase()} · ${escapeHtml(item.id)} · ${formatNumber(item.samples)} frames</option>`).join("");
  if (previous && [...select.options].some((option) => option.value === previous)) select.value = previous;
  $("#emptyExplorer").hidden = sequences.length > 0; $("#visualGrid").hidden = sequences.length === 0;
  renderEvaluationOverlay();
  if (!sequences.length) { state.detail = null; state.frame = null; state.prediction = null; return; }
  const chosen = selectedSequence();
  if (!state.detail || state.detail.version !== chosen.version || state.detail.split !== chosen.split || state.detail.id !== chosen.id) loadSequence();
}

async function loadSequence() {
  if (!$("#sequenceSelect").value) return;
  stopPlayback();
  const selected = selectedSequence();
  try {
    state.detail = await api(`/api/sequence?version=${encodeURIComponent(selected.version)}&split=${encodeURIComponent(selected.split)}&id=${encodeURIComponent(selected.id)}`);
    $("#frameSlider").max = Math.max(0, state.detail.samples - 1); $("#frameNumber").max = Math.max(0, state.detail.samples - 1);
    $("#frameSlider").value = $("#frameNumber").value = 0;
    renderSequenceDetail();
    await loadFrame(0);
  } catch (error) { toast(error.message, true); }
}

function renderSequenceDetail() {
  const detail = state.detail, target = detail.target;
  $("#sequenceSplit").textContent = detail.split.toUpperCase();
  $("#sensorOverview").innerHTML = detail.availability.map((availability, index) => {
    const distance = detail.distance_means?.[index];
    return `<div class="sensor-row" style="--sensor-color:${SENSOR_COLORS[index + 1]}"><b>RSU ${index + 1}</b><div class="coverage-track"><i style="width:${Math.max(0, Math.min(100, availability * 100))}%"></i></div><span>${(availability * 100).toFixed(0)}% · ${distance == null ? "—" : `${distance.toFixed(0)}m`}</span></div>`;
  }).join("");
  const rows = [
    ["Samples", formatNumber(detail.samples)], ["Scan rays", formatNumber(detail.scan_points)], ["RSU sensors", detail.rsu_count],
    ["Acceleration", `${target.accel_min.toFixed(3)} … ${target.accel_max.toFixed(3)}`], ["Accel mean", target.accel_mean.toFixed(3)],
    ["Steering", `${target.steer_min.toFixed(3)} … ${target.steer_max.toFixed(3)}`], ["Steer mean", target.steer_mean.toFixed(4)],
  ];
  $("#sequenceDetails").innerHTML = rows.map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`).join("");
  drawHistogram($("#steerHistogramCanvas"), target.steer_histogram, target.steer_edges, "steer");
  drawHistogram($("#accelHistogramCanvas"), target.accel_histogram, target.accel_edges, "accel");
}

async function loadFrame(index) {
  const selected = selectedSequence();
  const safeIndex = Math.max(0, Math.min(Number(index) || 0, Number($("#frameSlider").max)));
  try {
    state.frame = await api(`/api/frame?version=${encodeURIComponent(selected.version)}&split=${encodeURIComponent(selected.split)}&id=${encodeURIComponent(selected.id)}&index=${safeIndex}`);
    state.prediction = null;
    const evaluation = $("#evaluationOverlay").value;
    if (evaluation) {
      state.prediction = await api(`/api/prediction?evaluation=${encodeURIComponent(evaluation)}&id=${encodeURIComponent(selected.id)}&index=${safeIndex}`);
    }
    $("#frameSlider").value = $("#frameNumber").value = safeIndex;
    $("#frameLabel").textContent = `frame ${safeIndex + 1} / ${state.frame.samples}`;
    $("#frameSteer").textContent = state.frame.steering.toFixed(4); $("#frameAccel").textContent = state.frame.acceleration.toFixed(3);
    $("#frameAvailable").textContent = `${state.frame.mask.filter(Boolean).length} / ${state.frame.mask.length}`;
    $("#frameMode").textContent = state.prediction?.available ? `${state.prediction.selected_mode + 1} · ${(state.prediction.mode_probabilities[state.prediction.selected_mode] * 100).toFixed(0)}%` : "—";
    $("#predictedAccel").textContent = state.prediction?.available ? state.prediction.control[0].toFixed(3) : "—";
    renderSensorPills(); renderSensorInspector(); renderLegend(); drawVisualization();
  } catch (error) { stopPlayback(); toast(error.message, true); }
}

function sensorName(index) { return index === 0 ? "EGO" : `RSU ${index}`; }
function sensorAvailable(index) { return index === 0 || Boolean(state.frame?.mask[index - 1]); }
function renderSensorPills() {
  const labels = ["Virtual Scan", ...state.frame.rsus.map((_, index) => `curve_${String(index + 1).padStart(2, "0")}`)];
  $("#sensorPills").innerHTML = labels.map((subtitle, index) => {
    const available = sensorAvailable(index);
    return `<button type="button" class="${index === state.sensor ? "active " : ""}${available ? "" : "unavailable"}" data-sensor="${index}" style="--sensor-color:${SENSOR_COLORS[index]}"><strong>${sensorName(index)}</strong><small>${available ? subtitle : "not synchronized"}</small></button>`;
  }).join("");
  $$('[data-sensor]').forEach((button) => button.addEventListener("click", () => {
    state.sensor = Number(button.dataset.sensor); setView("sensor"); renderSensorPills(); renderSensorInspector(); renderLegend(); drawVisualization();
  }));
}
function renderSensorInspector() {
  const index = state.sensor, available = sensorAvailable(index), frame = state.frame;
  const status = $("#selectedSensorStatus");
  $("#selectedSensorName").textContent = sensorName(index); $("#selectedSensorDot").style.setProperty("--sensor-color", SENSOR_COLORS[index]);
  $("#selectedSensorSubtitle").textContent = index === 0 ? "Ego Virtual Scan" : `/rsu/curve_${String(index).padStart(2, "0")}/scan`;
  status.textContent = available ? "AVAILABLE" : "UNAVAILABLE"; status.className = `sensor-status ${available ? "available" : "unavailable"}`;
  let rows;
  if (index === 0) {
    rows = [["Coordinate", "ego frame"], ["Field of view", `${frame.sensor_fov_deg[0].toFixed(0)}°`], ["Rays", formatNumber(frame.ego.length)], ["Display range", `${frame.sensor_max_ranges[0].toFixed(0)} m`]];
  } else {
    const metadata = frame.meta[index - 1] || [0, 0, 0, 0, 0];
    rows = [["Distance", `${metadata[0].toFixed(2)} m`], ["Relative X", `${metadata[1].toFixed(2)} m`], ["Relative Y", `${metadata[2].toFixed(2)} m`], ["Relative yaw", `${(metadata[3] * 180 / Math.PI).toFixed(1)}°`], ["Scan age", `${metadata[4].toFixed(3)} s`], ["Field of view", `${frame.sensor_fov_deg[index].toFixed(0)}°`]];
  }
  $("#sensorInspector").innerHTML = rows.map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`).join("");
}
function renderLegend() {
  $("#mapLegend").innerHTML = SENSOR_COLORS.map((color, index) => `<span style="opacity:${sensorAvailable(index) ? 1 : .35}"><i style="color:${color};background:${color}"></i>${sensorName(index)}</span>`).join("");
}

function canvasContext(canvas) {
  const ratio = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(box.width * ratio)); canvas.height = Math.max(1, Math.floor(box.height * ratio));
  const context = canvas.getContext("2d"); context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: box.width, height: box.height };
}
function scanPoints(ranges, fovDegrees, maxRange) {
  const fov = fovDegrees * Math.PI / 180, denominator = Math.max(1, ranges.length - 1), points = [];
  ranges.forEach((range, index) => {
    if (!Number.isFinite(range) || range <= 0 || range >= maxRange) return;
    const angle = -fov / 2 + fov * index / denominator;
    points.push([Math.cos(angle) * range, Math.sin(angle) * range]);
  });
  return points;
}
function drawMapGrid(context, width, height, centerX, centerY, scale, range) {
  context.strokeStyle = "#26312e"; context.lineWidth = 1; context.fillStyle = "#66736e"; context.font = "10px ui-monospace";
  for (const ratio of [.25, .5, .75, 1]) {
    const radius = range * scale * ratio; context.beginPath(); context.arc(centerX, centerY, radius, 0, Math.PI * 2); context.stroke();
    context.fillText(`${(range * ratio).toFixed(0)}m`, centerX + 5, centerY - radius + 12);
  }
  context.strokeStyle = "#34413d"; context.beginPath(); context.moveTo(centerX, 14); context.lineTo(centerX, height - 14); context.moveTo(14, centerY); context.lineTo(width - 14, centerY); context.stroke();
  context.fillStyle = "#66736e"; context.fillText("FRONT", centerX + 7, 25);
}
function plotPoints(context, points, centerX, centerY, scale, color, alpha = 1, size = 2.1) {
  context.fillStyle = color; context.globalAlpha = alpha;
  for (const [forward, left] of points) context.fillRect(centerX - left * scale - size / 2, centerY - forward * scale - size / 2, size, size);
  context.globalAlpha = 1;
}
function drawVehicle(context, x, y, yaw, scale, color, selected = false) {
  context.save(); context.translate(x, y); context.rotate(-yaw); context.strokeStyle = color; context.fillStyle = `${color}33`; context.lineWidth = selected ? 2.5 : 1.5;
  context.beginPath(); context.rect(-.75 * scale, -1.2 * scale, 1.5 * scale, 2.4 * scale); context.fill(); context.stroke();
  context.beginPath(); context.moveTo(0, -1.2 * scale); context.lineTo(0, -2.1 * scale); context.stroke(); context.restore();
}
function drawTrajectory(context, points, centerX, centerY, scale, color, width = 2, dashed = false) {
  if (!points?.length) return;
  context.save(); context.strokeStyle = color; context.lineWidth = width; context.lineJoin = "round"; context.lineCap = "round";
  if (dashed) context.setLineDash([7, 5]);
  context.beginPath(); context.moveTo(centerX, centerY);
  points.forEach((point) => context.lineTo(centerX - point[1] * scale, centerY - point[0] * scale));
  context.stroke(); context.restore();
}
function drawPredictionOverlay(context, centerX, centerY, scale) {
  const prediction = state.prediction;
  if (!prediction?.available) return;
  prediction.trajectories.forEach((trajectory, mode) => {
    const selected = mode === prediction.selected_mode;
    const hue = 165 + mode * 48;
    drawTrajectory(context, trajectory, centerX, centerY, scale, `hsla(${hue},90%,65%,${selected ? 1 : .32})`, selected ? 4 : 1.5);
  });
  drawTrajectory(context, prediction.target_trajectory, centerX, centerY, scale, "#ffffff", 2.2, true);
}
function drawFusionMap(context, width, height) {
  const frame = state.frame, validDistances = frame.meta.filter((_, index) => frame.mask[index]).map((metadata) => Math.abs(metadata[0])).filter(Number.isFinite);
  const viewRange = Math.max(45, Math.min(90, Math.max(0, ...validDistances) + 24));
  const centerX = width / 2, centerY = height * .54, scale = Math.min(width * .44, height * .46) / viewRange;
  drawMapGrid(context, width, height, centerX, centerY, scale, viewRange);
  plotPoints(context, scanPoints(frame.ego, frame.sensor_fov_deg[0], frame.sensor_max_ranges[0]), centerX, centerY, scale, SENSOR_COLORS[0], state.sensor === 0 ? 1 : .55, state.sensor === 0 ? 2.7 : 2);
  frame.rsus.forEach((ranges, rsuIndex) => {
    if (!frame.mask[rsuIndex]) return;
    const metadata = frame.meta[rsuIndex], cos = Math.cos(metadata[3]), sin = Math.sin(metadata[3]);
    const transformed = scanPoints(ranges, frame.sensor_fov_deg[rsuIndex + 1], frame.sensor_max_ranges[rsuIndex + 1]).map(([x, y]) => [metadata[1] + cos * x - sin * y, metadata[2] + sin * x + cos * y]);
    const selected = state.sensor === rsuIndex + 1;
    plotPoints(context, transformed, centerX, centerY, scale, SENSOR_COLORS[rsuIndex + 1], selected ? 1 : .48, selected ? 3 : 2);
    drawVehicle(context, centerX - metadata[2] * scale, centerY - metadata[1] * scale, metadata[3], Math.max(2.2, scale), SENSOR_COLORS[rsuIndex + 1], selected);
  });
  drawVehicle(context, centerX, centerY, 0, Math.max(2.5, scale), SENSOR_COLORS[0], state.sensor === 0);
  drawPredictionOverlay(context, centerX, centerY, scale);
  const missingPose = frame.meta.every((metadata) => Math.abs(metadata[0]) < 1e-6 && Math.abs(metadata[1]) < 1e-6 && Math.abs(metadata[2]) < 1e-6);
  $("#canvasHint").textContent = missingPose ? "この旧データには自己位置由来のRSU配置がありません" : "EGO座標へRSU点群を変換して重ねています";
}
function drawSensorView(context, width, height) {
  const frame = state.frame, index = state.sensor, ranges = index === 0 ? frame.ego : frame.rsus[index - 1];
  const maxRange = frame.sensor_max_ranges[index], fov = frame.sensor_fov_deg[index] * Math.PI / 180;
  const centerX = width / 2, centerY = height * .57, scale = Math.min(width * .43, height * .45) / maxRange;
  context.strokeStyle = "#26312e"; context.fillStyle = "#66736e"; context.font = "10px ui-monospace";
  for (const ratio of [.25, .5, .75, 1]) {
    const radius = maxRange * scale * ratio; context.beginPath(); context.arc(centerX, centerY, radius, -Math.PI / 2 - fov / 2, -Math.PI / 2 + fov / 2); context.stroke(); context.fillText(`${(maxRange * ratio).toFixed(0)}m`, centerX + 5, centerY - radius + 12);
  }
  context.strokeStyle = "#35423e";
  for (const edge of [-fov / 2, 0, fov / 2]) { context.beginPath(); context.moveTo(centerX, centerY); context.lineTo(centerX - Math.sin(edge) * maxRange * scale, centerY - Math.cos(edge) * maxRange * scale); context.stroke(); }
  plotPoints(context, scanPoints(ranges, frame.sensor_fov_deg[index], maxRange), centerX, centerY, scale, SENSOR_COLORS[index], sensorAvailable(index) ? 1 : .25, 2.8);
  drawVehicle(context, centerX, centerY, 0, Math.max(2.7, scale), SENSOR_COLORS[index], true);
  if (index === 0) drawPredictionOverlay(context, centerX, centerY, scale);
  $("#canvasHint").textContent = sensorAvailable(index) ? `${sensorName(index)}のローカル座標表示` : `${sensorName(index)}はこのframeで同期できていません`;
}
function drawVisualization() {
  if (!state.frame) return;
  const { context, width, height } = canvasContext($("#fusionCanvas")); context.clearRect(0, 0, width, height);
  state.view === "map" ? drawFusionMap(context, width, height) : drawSensorView(context, width, height);
}
function drawHistogram(canvas, values, edges, kind) {
  if (!values?.length || !edges?.length) return;
  const { context, width, height } = canvasContext(canvas), pad = { left: 29, right: 8, top: 8, bottom: 21 };
  const plotWidth = width - pad.left - pad.right, plotHeight = height - pad.top - pad.bottom, maximum = Math.max(...values, 1), barWidth = plotWidth / values.length;
  context.clearRect(0, 0, width, height); context.strokeStyle = "#35403d"; context.beginPath(); context.moveTo(pad.left, pad.top); context.lineTo(pad.left, pad.top + plotHeight); context.lineTo(pad.left + plotWidth, pad.top + plotHeight); context.stroke();
  values.forEach((value, index) => {
    const barHeight = value / maximum * plotHeight, center = (edges[index] + edges[index + 1]) / 2;
    context.fillStyle = kind === "accel" ? (center > .8 ? "#c9fa3f" : "#54e6d0") : (Math.abs(center) <= .02 ? "#c9fa3f" : center < 0 ? "#54e6d0" : "#ff9d52");
    context.fillRect(pad.left + index * barWidth + 1, pad.top + plotHeight - barHeight, Math.max(1, barWidth - 2), barHeight);
  });
  context.fillStyle = "#82908a"; context.font = "9px ui-monospace"; context.textAlign = "center";
  context.fillText(edges[0].toFixed(2), pad.left, height - 5); context.fillText(edges.at(-1).toFixed(2), pad.left + plotWidth, height - 5);
}

function setView(view) {
  state.view = view; $$('[data-view]').forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $("#mapLegend").hidden = view !== "map";
  $("#viewTitle").textContent = view === "map" ? "Ego-frame Fusion Map" : `${sensorName(state.sensor)} Scan`;
  $("#frameView").textContent = view === "map" ? "FUSION" : sensorName(state.sensor); drawVisualization();
}
function startPlayback() {
  state.playing = true; $("#playButton").textContent = "Ⅱ";
  state.timer = setInterval(() => loadFrame((Number($("#frameSlider").value) + 1) % (Number($("#frameSlider").max) + 1)), 120);
}
function stopPlayback() { clearInterval(state.timer); state.timer = null; state.playing = false; $("#playButton").textContent = "▶"; }

function renderRecordings() {
  const recordings = state.overview.recordings;
  $("#recordingList").innerHTML = recordings.length ? recordings.map((item) => `<div class="recording-row"><div><b>${escapeHtml(item.name)}</b><small>${escapeHtml(item.id)}</small></div><span>${formatBytes(item.size_bytes)}</span><label><input type="checkbox" data-split="train" value="${escapeHtml(item.id)}"></label><label><input type="checkbox" data-split="val" value="${escapeHtml(item.id)}"></label></div>`).join("") : '<div class="empty">録画がありません。</div>';
}

const collectionCategoryLabels = { free_optimal: "通常・最適", free_diverse: "通常・多様", follow: "追従", pass_left: "左追い越し", pass_right: "右追い越し", abort: "追い越し断念", recovery: "復帰操作", other: "その他" };
const collectionOutcomeLabels = { success: "成功", failure: "失敗", review: "要確認" };

function loadCollectionForm(recording) {
  const annotation = recording?.annotation || null;
  state.collectionLoaded = recording?.id || null;
  state.collectionStamp = annotation?.updated_at || null;
  $("#collectionTitle").textContent = recording ? recording.name : "録画を選択してください";
  $("#collectionCategory").value = annotation?.category || "free_optimal";
  $("#collectionOutcome").value = annotation?.outcome || "success";
  $("#collectionQuality").value = annotation?.quality || "accepted";
  $("#collectionVersions").value = (annotation?.dataset_versions || []).join(", ");
  $("#collectionNotes").value = annotation?.notes || "";
  const saved = $("#collectionSavedState");
  saved.textContent = annotation ? `保存済み · ${annotation.updated_at || ""}` : "未分類";
  saved.className = annotation ? "saved" : "unsaved";
  $$("#collectionForm input, #collectionForm select, #collectionForm textarea, #collectionForm button").forEach((element) => { element.disabled = !recording; });
}

function selectCollectionRecording(recordingId) {
  state.collectionSelected = recordingId;
  state.collectionLoaded = null;
  renderCollection();
}

function renderCollection() {
  const recordings = state.overview?.recordings || [];
  const newest = recordings[0]?.id || null;
  if (state.collectionLatest !== newest) {
    state.collectionLatest = newest;
    state.collectionSelected = newest;
    state.collectionLoaded = null;
  }
  if (!recordings.some((item) => item.id === state.collectionSelected)) state.collectionSelected = newest;
  const select = $("#collectionRecording");
  select.innerHTML = recordings.length ? recordings.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)}${item.annotation ? ` · ${escapeHtml(collectionOutcomeLabels[item.annotation.outcome] || item.annotation.outcome)}` : " · 未分類"}</option>`).join("") : '<option value="">録画がありません</option>';
  select.value = state.collectionSelected || "";
  $("#collectionWatchState").textContent = recordings.length ? `最新 ${formatTime(recordings[0].modified_at)}` : "L1停止後に自動表示";
  $("#collectionRecent").innerHTML = recordings.length ? recordings.slice(0, 12).map((item) => {
    const annotation = item.annotation;
    const label = annotation ? `${collectionCategoryLabels[annotation.category] || annotation.category} · ${collectionOutcomeLabels[annotation.outcome] || annotation.outcome}` : "未分類";
    return `<button type="button" class="${item.id === state.collectionSelected ? "active" : ""}" data-collection-recording="${escapeHtml(item.id)}"><b>${escapeHtml(item.name)}</b><span class="collection-badge ${annotation ? "saved" : ""}">${escapeHtml(label)}</span><small>${escapeHtml(item.id)} · ${formatTime(item.modified_at)}</small></button>`;
  }).join("") : '<div class="empty">Bag Managerの録画待ちです。</div>';
  $$('[data-collection-recording]').forEach((button) => button.addEventListener("click", () => selectCollectionRecording(button.dataset.collectionRecording)));
  const selected = recordings.find((item) => item.id === state.collectionSelected) || null;
  const stamp = selected?.annotation?.updated_at || null;
  if (state.collectionLoaded !== selected?.id || state.collectionStamp !== stamp) loadCollectionForm(selected);
}

async function saveCollectionAnnotation(event) {
  event.preventDefault();
  if (!state.collectionSelected) return;
  const payload = {
    recording_id: state.collectionSelected,
    category: $("#collectionCategory").value,
    outcome: $("#collectionOutcome").value,
    quality: $("#collectionQuality").value,
    dataset_versions: $("#collectionVersions").value.split(",").map((value) => value.trim()).filter(Boolean),
    notes: $("#collectionNotes").value,
  };
  try {
    const result = await api("/api/recordings/annotate", { method: "POST", body: JSON.stringify(payload) });
    const recording = state.overview.recordings.find((item) => item.id === state.collectionSelected);
    if (recording) recording.annotation = result.annotation;
    state.collectionLoaded = null;
    renderCollection(); renderRecordings();
    toast("分類とメモを保存しました");
  } catch (error) { toast(error.message, true); }
}

async function pollLatestRecording() {
  if (state.collectionPolling) return;
  state.collectionPolling = true;
  try {
    const result = await api("/api/recordings/latest");
    const latestId = result.recording?.id || null;
    if (latestId !== state.collectionLatest) {
      await refreshOverview();
      if (latestId) { showTab("collection"); toast("新しい録画を検出しました。分類を入力できます"); }
    }
  } catch (_) {
    // The normal overview refresh reports connection errors; keep this poll quiet.
  } finally { state.collectionPolling = false; }
}
function renderTraining() {
  const inputMode = $("#trainingInputMode").value;
  const ready = (item) => item.valid && item.trajectory_ready && (inputMode !== "bev" || item.bev_ready);
  const train = versionSequences("train").filter(ready), validation = versionSequences("val").filter(ready);
  const encoder = inputMode === "bev" ? "Semantic BEV · 2D CNN + GRU" : "Ego/RSU Scan · 1D CNN + GRU";
  $("#readiness").innerHTML = `<span>Train <b class="${train.length ? "ok" : "warn"}">${train.length ? formatNumber(train.reduce((sum, item) => sum + item.samples, 0)) : "再前処理が必要"}</b></span><span>Validation <b class="${validation.length ? "ok" : "warn"}">${validation.length ? formatNumber(validation.reduce((sum, item) => sum + item.samples, 0)) : "再前処理が必要"}</b></span><span>Input <b class="ok">${encoder}</b></span><span>Outputs <b class="ok">Trajectory + Speed + Control</b></span>`;
  $("#trainingForm button[type=submit]").disabled = !train.length || !validation.length;
  $("#topK").disabled = inputMode === "bev"; $("#distanceDecay").disabled = inputMode === "bev";
  const checkpoints = state.overview.checkpoints.filter((item) => item.best);
  $("#pretrained").innerHTML = '<option value="">使用しない</option>' + checkpoints.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)}</option>`).join("");
  $("#checkpointList").innerHTML = checkpoints.slice(0, 8).map((item) => `<div>${escapeHtml(item.id)}</div>`).join("");
}
function renderTrajectorySpec() {
  const steps = Number($("#trajectorySteps").value), dt = Number($("#trajectoryDt").value);
  const anchors = Number($("#trajectoryAnchors").value), horizon = steps * dt;
  const valid = Number.isFinite(horizon) && horizon > 0 && Number.isFinite(anchors) && anchors > 0;
  $("#trajectorySpec").textContent = valid
    ? `${horizon.toFixed(2)}秒先まで · Waypoint ${dt.toFixed(2)}秒間隔 · Bezierアンカー ${(horizon / anchors).toFixed(2)}秒間隔。ADE（平均）とFDE（終端）を学習します。`
    : "軌跡設定を確認してください。";
}
async function submitPreprocess(event) {
  event.preventDefault(); const selectedIds = (split) => $$(`input[data-split="${split}"]:checked`).map((input) => input.value);
  const version = $("#newVersion").value.trim() || $("#preprocessVersion").value;
  try {
    const job = await api("/api/jobs/preprocess", { method: "POST", body: JSON.stringify({ dataset_version: version, train: selectedIds("train"), val: selectedIds("val"), input_mode: $("#preprocessInputMode").value, existing_policy: $("#existingPolicy").value, ego_scan_topic: $("#egoScanTopic").value, max_sync_dt: Number($("#syncDelta").value), scan_dim: 1080 }) });
    state.selectedJob = job.id; state.activeVersion = version; localStorage.setItem("rsuFusionVersion", version); showTab("jobs"); toast("RSU前処理を開始しました"); await refreshOverview();
  } catch (error) { toast(error.message, true); }
}
async function submitTraining(event) {
  event.preventDefault();
  try {
    const job = await api("/api/jobs/train", { method: "POST", body: JSON.stringify({ dataset_version: state.activeVersion, input_mode: $("#trainingInputMode").value, history_len: Number($("#historyLen").value), top_k_rsus: Number($("#topK").value), distance_decay_m: Number($("#distanceDecay").value), trajectory_modes: Number($("#trajectoryModes").value), trajectory_steps: Number($("#trajectorySteps").value), trajectory_dt: Number($("#trajectoryDt").value), trajectory_anchors: Number($("#trajectoryAnchors").value), ade_weight: Number($("#adeWeight").value), fde_weight: Number($("#fdeWeight").value), pretrained: $("#pretrained").value, epochs: Number($("#epochs").value), batch_size: Number($("#batchSize").value), learning_rate: Number($("#learningRate").value), patience: Number($("#patience").value), workers: Number($("#workers").value) }) });
    state.selectedJob = job.id; showTab("jobs"); toast("RSU学習を開始しました"); await refreshOverview();
  } catch (error) { toast(error.message, true); }
}
function renderEvaluationOverlay() {
  const select = $("#evaluationOverlay");
  if (!select || !state.overview) return;
  const previous = select.value;
  const selected = selectedSequence();
  const evaluations = state.overview.evaluations.filter((item) => item.version === state.activeVersion && (!selected.split || item.split === selected.split));
  select.innerHTML = '<option value="">なし</option>' + evaluations.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)} · ADE ${Number(item.metrics.ade_m || 0).toFixed(2)}m</option>`).join("");
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
}
function renderEvaluations() {
  const checkpoints = state.overview.checkpoints;
  $("#evaluationCheckpoint").innerHTML = checkpoints.length ? checkpoints.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)}</option>`).join("") : '<option value="">checkpointなし</option>';
  const evaluations = state.overview.evaluations.filter((item) => item.version === state.activeVersion);
  if (!evaluations.some((item) => item.id === state.selectedEvaluation)) {
    state.selectedEvaluation = evaluations[0]?.id || null;
    state.course = null; state.courseFrame = null;
  }
  $("#evaluationList").innerHTML = evaluations.length ? evaluations.map((item) => {
    const metric = item.metrics || {};
    return `<button type="button" class="${item.id === state.selectedEvaluation ? "active" : ""}" data-evaluation="${escapeHtml(item.id)}"><b>${escapeHtml(item.split.toUpperCase())} · ${escapeHtml(item.id)}</b><span>ADE <strong>${Number(metric.ade_m || 0).toFixed(3)} m</strong> · FDE <strong>${Number(metric.fde_m || 0).toFixed(3)} m</strong></span><span>Speed <strong>${Number(metric.speed_mae_mps || 0).toFixed(3)} m/s</strong> · Accel <strong>${Number(metric.acceleration_mae || 0).toFixed(3)}</strong> · Steer <strong>${Number(metric.steering_mae_rad || 0).toFixed(4)}</strong></span></button>`;
  }).join("") : '<div class="empty">このVersionの評価結果はありません。</div>';
  $$('[data-evaluation]').forEach((button) => button.addEventListener("click", async () => {
    state.selectedEvaluation = button.dataset.evaluation;
    $$('[data-evaluation]').forEach((item) => item.classList.toggle("active", item.dataset.evaluation === state.selectedEvaluation));
    await loadEvaluationCourse();
  }));
  renderEvaluationOverlay();
  if (state.selectedEvaluation && state.course?.evaluation !== state.selectedEvaluation) loadEvaluationCourse();
}

async function loadEvaluationCourse(sequence = "") {
  if (!state.selectedEvaluation) return;
  stopCoursePlayback();
  const request = ++state.courseRequest;
  $("#courseHint").textContent = "コース評価を読み込み中…";
  try {
    const course = await api(`/api/evaluation-course?evaluation=${encodeURIComponent(state.selectedEvaluation)}&sequence=${encodeURIComponent(sequence)}`);
    if (request !== state.courseRequest) return;
    state.course = course; state.courseFrame = null;
    const trajectory = course.trajectory || {}, horizon = Number(trajectory.steps || 0) * Number(trajectory.dt || 0);
    const decoder = trajectory.architecture === "trajectory_bezier_v2" ? `Bezier ${trajectory.anchor_count} anchors` : "Legacy direct points";
    $("#courseEvaluationName").textContent = `${course.evaluation} · ${formatNumber(course.sample_indices.length)} predictions · ${horizon.toFixed(2)}s / ${Number(trajectory.dt || 0).toFixed(2)}s · ${decoder}`;
    $("#courseSequence").innerHTML = course.sequence_options.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)} · ADE ${Number(item.metrics?.ade_m || 0).toFixed(3)}m</option>`).join("");
    $("#courseSequence").value = course.sequence;
    const maximum = Math.max(0, course.sample_indices.length - 1);
    $("#courseFrameSlider").max = $("#courseFrameNumber").max = maximum;
    $("#courseFrameSlider").value = $("#courseFrameNumber").value = 0;
    $("#courseHint").textContent = "点をクリックすると、その位置の予測候補を表示します";
    drawCourseMap();
    await loadCourseFrame(0);
  } catch (error) {
    if (request === state.courseRequest) {
      $("#courseHint").textContent = `可視化を読み込めません: ${error.message}`;
      toast(error.message, true);
    }
  }
}

async function loadCourseFrame(position) {
  if (!state.course?.sample_indices?.length) return;
  const safePosition = Math.max(0, Math.min(Number(position) || 0, state.course.sample_indices.length - 1));
  const sampleIndex = state.course.sample_indices[safePosition];
  const request = ++state.courseRequest;
  try {
    const frame = await api(`/api/prediction?evaluation=${encodeURIComponent(state.course.evaluation)}&id=${encodeURIComponent(state.course.sequence)}&index=${sampleIndex}`);
    if (request !== state.courseRequest) return;
    state.courseFrame = frame;
    $("#courseFrameSlider").value = $("#courseFrameNumber").value = safePosition;
    $("#courseFrameLabel").textContent = `${formatNumber(sampleIndex)} · ${safePosition + 1}/${formatNumber(state.course.sample_indices.length)}`;
    $("#courseMode").textContent = `${frame.selected_mode + 1} · ${(frame.mode_probabilities[frame.selected_mode] * 100).toFixed(0)}%`;
    $("#courseAde").textContent = Number(frame.frame_metrics.ade_m).toFixed(3);
    $("#courseFde").textContent = Number(frame.frame_metrics.fde_m).toFixed(3);
    $("#courseAccel").textContent = Number(frame.control[0]).toFixed(3);
    $("#courseTargetAccel").textContent = Number(frame.target_control[0]).toFixed(3);
    drawCourseMap();
  } catch (error) { stopCoursePlayback(); toast(error.message, true); }
}

function mapBounds(course, frame) {
  const points = [...course.lanes.flat(), ...course.route];
  if (frame?.available) points.push(...frame.map_trajectories.flat(), ...frame.map_target_trajectory, frame.ego_pose);
  if (!points.length) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  const xs = points.map((point) => point[0]), ys = points.map((point) => point[1]);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}
function mapPolyline(context, points, project, color, width = 1, dashed = false, alpha = 1) {
  if (!points?.length) return;
  context.save(); context.strokeStyle = color; context.lineWidth = width; context.globalAlpha = alpha; context.lineJoin = "round"; context.lineCap = "round";
  if (dashed) context.setLineDash([7, 5]);
  const first = project(points[0]); context.beginPath(); context.moveTo(first[0], first[1]);
  points.slice(1).forEach((point) => { const screen = project(point); context.lineTo(screen[0], screen[1]); });
  context.stroke(); context.restore();
}
function mapWaypoints(context, points, project, color, radius = 3, alpha = 1) {
  context.save(); context.fillStyle = color; context.globalAlpha = alpha;
  points.forEach((point) => {
    const screen = project(point); context.beginPath();
    context.arc(screen[0], screen[1], radius, 0, Math.PI * 2); context.fill();
  });
  context.restore();
}
function drawCourseMap() {
  const canvas = $("#courseMapCanvas"), course = state.course;
  const { context, width, height } = canvasContext(canvas); context.clearRect(0, 0, width, height);
  if (!course) return;
  const bounds = mapBounds(course, state.courseFrame), padding = 34;
  const spanX = Math.max(1, bounds.maxX - bounds.minX), spanY = Math.max(1, bounds.maxY - bounds.minY);
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const offsetX = (width - spanX * scale) / 2, offsetY = (height - spanY * scale) / 2;
  const project = (point) => [offsetX + (point[0] - bounds.minX) * scale, height - offsetY - (point[1] - bounds.minY) * scale];
  state.courseTransform = { project, scale };
  context.strokeStyle = "#1c2925"; context.lineWidth = 1;
  for (let x = 0; x <= width; x += 50) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = 0; y <= height; y += 50) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  course.lanes.forEach((lane) => mapPolyline(context, lane, project, "#53615c", 1.2, false, .7));
  mapPolyline(context, course.route, project, "#54e6d0", 1.4, false, .5);
  course.heat_points.forEach((point) => {
    const screen = project(point), ratio = Math.max(0, Math.min(1, point[2] / course.error_scale_m));
    context.fillStyle = `hsla(${165 * (1 - ratio)},90%,60%,.72)`;
    context.beginPath(); context.arc(screen[0], screen[1], 2.2 + ratio * 2.2, 0, Math.PI * 2); context.fill();
  });
  const frame = state.courseFrame;
  if (frame?.available) {
    frame.map_trajectories.forEach((trajectory, mode) => {
      const selected = mode === frame.selected_mode;
      const color = `hsl(${165 + mode * 48},90%,65%)`;
      mapPolyline(context, trajectory, project, color, selected ? 4 : 1.4, false, selected ? 1 : .3);
      if (selected) mapWaypoints(context, trajectory, project, color, 3.2, 1);
    });
    mapPolyline(context, frame.map_target_trajectory, project, "#fff", 2.3, true, .95);
    mapWaypoints(context, frame.map_target_trajectory, project, "#fff", 2.4, .9);
    const ego = project(frame.ego_pose); context.fillStyle = "#c9fa3f"; context.beginPath(); context.arc(ego[0], ego[1], 6, 0, Math.PI * 2); context.fill();
  }
}
function startCoursePlayback() {
  if (!state.course?.sample_indices?.length) return;
  state.coursePlaying = true; $("#coursePlayButton").textContent = "Ⅱ";
  state.courseTimer = setInterval(() => loadCourseFrame((Number($("#courseFrameSlider").value) + 1) % state.course.sample_indices.length), 180);
}
function stopCoursePlayback() { clearInterval(state.courseTimer); state.courseTimer = null; state.coursePlaying = false; $("#coursePlayButton").textContent = "▶"; }
async function submitEvaluation(event) {
  event.preventDefault();
  try {
    const job = await api("/api/jobs/evaluate", { method: "POST", body: JSON.stringify({ dataset_version: state.activeVersion, checkpoint: $("#evaluationCheckpoint").value, split: $("#evaluationSplit").value, device: $("#evaluationDevice").value, batch_size: Number($("#evaluationBatch").value) }) });
    state.selectedJob = job.id; showTab("jobs"); toast("オフライン評価を開始しました"); await refreshOverview();
  } catch (error) { toast(error.message, true); }
}
function renderJobs() {
  const jobs = state.overview.jobs;
  if (!jobs.length) { $("#jobList").innerHTML = '<div class="empty">ジョブはありません。</div>'; return; }
  if (!state.selectedJob || !jobs.some((job) => job.id === state.selectedJob)) state.selectedJob = jobs[0].id;
  const labels = { train: "学習", preprocess: "前処理", evaluate: "評価" };
  $("#jobList").innerHTML = jobs.map((job) => `<button class="job-item ${job.id === state.selectedJob ? "active" : ""}" data-job="${job.id}"><b>${labels[job.kind] || job.kind}</b><span class="status ${job.status}">${job.status}</span><small>${job.id}${job.pid ? ` · PID ${job.pid}` : ""}</small></button>`).join("");
  $$('[data-job]').forEach((button) => button.addEventListener("click", () => { state.selectedJob = button.dataset.job; renderJobs(); }));
  const job = jobs.find((item) => item.id === state.selectedJob), log = $("#jobLog"), follow = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  $("#logTitle").textContent = `${job.kind} · ${job.id} · ${job.status}`; log.textContent = job.log || "開始待ち…"; if (follow) log.scrollTop = log.scrollHeight;
  $("#cancelJob").hidden = !["queued", "running"].includes(job.status);
}
function showTab(id) {
  $$(".tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === id));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === id));
  if (id === "explorer" && state.frame) setTimeout(() => { drawVisualization(); renderSequenceDetail(); }, 0);
  if (id === "evaluation" && state.course) setTimeout(drawCourseMap, 0);
}

$$(".tabs button").forEach((button) => button.addEventListener("click", () => { history.replaceState(null, "", `${location.pathname}${location.search}#${button.dataset.tab}`); showTab(button.dataset.tab); }));
$$('[data-view]').forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
$("#datasetVersion").addEventListener("change", (event) => { state.activeVersion = event.target.value; localStorage.setItem("rsuFusionVersion", state.activeVersion); state.detail = null; state.frame = null; state.prediction = null; state.selectedEvaluation = null; state.course = null; state.courseFrame = null; stopCoursePlayback(); renderVersions(); renderMetrics(); renderSequences(); renderTraining(); renderEvaluations(); });
$("#sequenceSelect").addEventListener("change", loadSequence); $("#frameSlider").addEventListener("input", (event) => loadFrame(event.target.value)); $("#frameNumber").addEventListener("change", (event) => loadFrame(event.target.value));
$("#playButton").addEventListener("click", () => state.playing ? stopPlayback() : startPlayback());
$("#preprocessForm").addEventListener("submit", submitPreprocess); $("#trainingForm").addEventListener("submit", submitTraining);
$("#collectionForm").addEventListener("submit", saveCollectionAnnotation);
$("#collectionRecording").addEventListener("change", (event) => selectCollectionRecording(event.target.value));
$("#trainingInputMode").addEventListener("change", renderTraining);
["#trajectorySteps", "#trajectoryDt", "#trajectoryAnchors"].forEach((selector) => $(selector).addEventListener("input", renderTrajectorySpec));
$("#evaluationForm").addEventListener("submit", submitEvaluation); $("#evaluationOverlay").addEventListener("change", () => loadFrame($("#frameSlider").value));
$("#courseSequence").addEventListener("change", (event) => loadEvaluationCourse(event.target.value));
$("#courseFrameSlider").addEventListener("input", (event) => loadCourseFrame(event.target.value));
$("#courseFrameNumber").addEventListener("change", (event) => loadCourseFrame(event.target.value));
$("#coursePlayButton").addEventListener("click", () => state.coursePlaying ? stopCoursePlayback() : startCoursePlayback());
$("#courseMapCanvas").addEventListener("click", (event) => {
  if (!state.course?.heat_points?.length || !state.courseTransform) return;
  const box = event.currentTarget.getBoundingClientRect(), x = event.clientX - box.left, y = event.clientY - box.top;
  let closest = null, closestDistance = Infinity;
  state.course.heat_points.forEach((point) => {
    const screen = state.courseTransform.project(point), distance = Math.hypot(screen[0] - x, screen[1] - y);
    if (distance < closestDistance) { closest = point; closestDistance = distance; }
  });
  if (closest && closestDistance <= 18) loadCourseFrame(closest[5]);
});
$("#refreshButton").addEventListener("click", refreshOverview); $("#cancelJob").addEventListener("click", async () => { await api(`/api/jobs/${state.selectedJob}/cancel`, { method: "POST", body: "{}" }); refreshOverview(); });
window.addEventListener("resize", () => { if (state.frame) { drawVisualization(); renderSequenceDetail(); } if (state.course) drawCourseMap(); });

$("#serverState").textContent = "UI READY · データ読込中";
const initialTab = location.hash.slice(1);
if (initialTab && document.getElementById(initialTab)?.classList.contains("tab-panel")) showTab(initialTab);
renderTrajectorySpec();
refreshOverview();
setInterval(() => { if (state.overview?.jobs.some((job) => ["queued", "running"].includes(job.status))) refreshOverview(); }, 1500);
setInterval(pollLatestRecording, 2000);
