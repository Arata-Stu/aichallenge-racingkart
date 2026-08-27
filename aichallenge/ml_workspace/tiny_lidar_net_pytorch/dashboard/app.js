const state = { overview: null, detail: null, frame: null, playing: false, timer: null, selectedJob: null, activeVersion: localStorage.getItem("tinyLidarDatasetVersion") || "default", collectionSelected: null, collectionLoaded: null, collectionStamp: null, collectionLatest: null, collectionPolling: false };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

function fmtNumber(value) { return new Intl.NumberFormat("ja-JP").format(value || 0); }
function fmtBytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}
function fmtTime(value) { return value ? new Date(value * 1000).toLocaleString("ja-JP") : "—"; }
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast show${error ? " error" : ""}`;
  clearTimeout(element._timer);
  element._timer = setTimeout(() => { element.className = "toast"; }, 3500);
}

async function refreshOverview(keepSelection = true) {
  const previous = keepSelection ? $("#sequenceSelect").value : "";
  try {
    state.overview = await api("/api/overview");
    $("#serverState").textContent = `ONLINE · PID ${state.overview.server.pid}`;
    $(".server-state").classList.add("online");
    renderVersions(); renderMetrics(); renderSequences(previous); renderRecordings(); renderCollection(); renderCheckpoints(); renderReadiness(); renderJobs();
  } catch (error) {
    $("#serverState").textContent = "OFFLINE";
    $(".server-state").classList.remove("online");
    toast(error.message, true);
  }
}

function renderVersions() {
  const globalSelect = $("#datasetVersion");
  const preprocessSelect = $("#preprocessVersion");
  const versions = state.overview.versions || [{ id: "default", train_samples: 0, val_samples: 0 }];
  const known = versions.some((item) => item.id === state.activeVersion);
  const items = known ? versions : [...versions, { id: state.activeVersion, train_samples: 0, val_samples: 0, pending: true }];
  const options = items.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)}${item.id === "default" ? " · 既存データ" : ""}${item.pending ? " · 作成待ち" : ""}</option>`).join("");
  globalSelect.innerHTML = options;
  preprocessSelect.innerHTML = options;
  globalSelect.value = state.activeVersion;
  preprocessSelect.value = state.activeVersion;
  const metadata = versions.find((item) => item.id === state.activeVersion);
  $("#activeVersionLabel").textContent = state.activeVersion;
  $("#trainingVersionLabel").textContent = state.activeVersion;
  $("#versionSummary").textContent = metadata ? `Train ${fmtNumber(metadata.train_samples)} · Val ${fmtNumber(metadata.val_samples)}` : "前処理を開始すると作成されます";
}

function switchDatasetVersion(version) {
  state.activeVersion = version || "default";
  localStorage.setItem("tinyLidarDatasetVersion", state.activeVersion);
  state.detail = null; state.frame = null; stopPlayback();
  $("#preprocessVersion").value = state.activeVersion;
  renderVersions(); renderMetrics(); renderSequences(""); renderReadiness();
}

function renderMetrics() {
  const data = state.overview;
  const train = data.sequences.filter((item) => item.version === state.activeVersion && item.split === "train");
  const val = data.sequences.filter((item) => item.version === state.activeVersion && item.split === "val");
  $("#recordingCount").textContent = fmtNumber(data.recordings.length);
  $("#trainSamples").textContent = fmtNumber(train.reduce((total, item) => total + item.samples, 0));
  $("#valSamples").textContent = fmtNumber(val.reduce((total, item) => total + item.samples, 0));
  $("#trainSequences").textContent = `${train.length} sequences`;
  $("#valSequences").textContent = `${val.length} sequences`;
  $("#checkpointCount").textContent = fmtNumber(data.checkpoints.length);
  const running = data.jobs.some((job) => ["queued", "running"].includes(job.status));
  $("#runningBadge").hidden = !running;
}

function renderSequences(previous) {
  const select = $("#sequenceSelect");
  const sequences = state.overview.sequences.filter((item) => item.version === state.activeVersion);
  select.innerHTML = sequences.map((item) => `<option value="${encodeURIComponent(item.split)}|${encodeURIComponent(item.id)}">${item.split.toUpperCase()} · ${escapeHtml(item.id)} · ${fmtNumber(item.samples)} frames</option>`).join("");
  $("#emptyExplorer").hidden = sequences.length > 0;
  $("#visualGrid").hidden = sequences.length === 0;
  if (!sequences.length) { state.detail = null; state.frame = null; return; }
  if (previous && [...select.options].some((option) => option.value === previous)) select.value = previous;
  const selected = selectedSequence();
  const selectedMetadata = sequences.find((item) => item.split === selected.split && item.id === selected.id);
  if (!state.detail || state.detail.version !== selected.version || state.detail.split !== selected.split || state.detail.id !== selected.id || state.detail.samples !== selectedMetadata?.samples) loadSelectedSequence();
}

function selectedSequence() {
  const [split = "", id = ""] = $("#sequenceSelect").value.split("|");
  return { version: state.activeVersion, split: decodeURIComponent(split), id: decodeURIComponent(id) };
}

async function loadSelectedSequence() {
  if (!$("#sequenceSelect").value) return;
  stopPlayback();
  const { version, split, id } = selectedSequence();
  try {
    state.detail = await api(`/api/sequence?version=${encodeURIComponent(version)}&split=${encodeURIComponent(split)}&id=${encodeURIComponent(id)}`);
    const slider = $("#frameSlider");
    slider.max = Math.max(0, state.detail.samples - 1);
    slider.value = 0;
    $("#frameNumber").max = slider.max;
    $("#frameNumber").value = 0;
    renderDetail();
    await loadFrame(0);
  } catch (error) { toast(error.message, true); }
}

function renderDetail() {
  const d = state.detail; const s = d.steering; const summary = d.summary || {};
  $("#sequenceSplit").textContent = d.split.toUpperCase();
  $("#distribution").innerHTML = [
    ["LEFT", s.left_ratio], ["STRAIGHT", s.straight_ratio], ["RIGHT", s.right_ratio],
  ].map(([label, value]) => `<span>${label}<b>${(value * 100).toFixed(1)}%</b></span>`).join("");
  const rows = [
    ["Samples", fmtNumber(d.samples)], ["Scan points", fmtNumber(d.scan_points)],
    ["Steer range", `${s.min.toFixed(3)} … ${s.max.toFixed(3)}`],
    ["Steer mean", s.mean.toFixed(4)], ["Acceleration mean", d.acceleration.mean.toFixed(3)],
    ["Sync mean", summary.sync_delta_mean_seconds == null ? "—" : `${Number(summary.sync_delta_mean_seconds).toFixed(4)} s`],
    ["Discarded scans", summary.discarded_scans == null ? "—" : fmtNumber(summary.discarded_scans)],
  ];
  $("#sequenceDetails").innerHTML = rows.map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`).join("");
  drawHistogram(s.histogram, s.edges);
}

async function loadFrame(index) {
  const { version, split, id } = selectedSequence();
  const safeIndex = Math.max(0, Math.min(Number(index) || 0, Number($("#frameSlider").max)));
  try {
    state.frame = await api(`/api/frame?version=${encodeURIComponent(version)}&split=${encodeURIComponent(split)}&id=${encodeURIComponent(id)}&index=${safeIndex}`);
    $("#frameSlider").value = safeIndex; $("#frameNumber").value = safeIndex;
    $("#frameLabel").textContent = `frame ${safeIndex + 1} / ${state.frame.samples}`;
    $("#frameSteer").textContent = state.frame.steering.toFixed(4);
    $("#frameAccel").textContent = state.frame.acceleration.toFixed(3);
    $("#framePoints").textContent = fmtNumber(state.frame.ranges.length);
    drawScan(state.frame);
  } catch (error) { stopPlayback(); toast(error.message, true); }
}

function canvasContext(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.floor(rect.width * ratio) || canvas.height !== Math.floor(rect.height * ratio)) {
    canvas.width = Math.floor(rect.width * ratio); canvas.height = Math.floor(rect.height * ratio);
  }
  const context = canvas.getContext("2d"); context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function drawScan(frame) {
  const { context: ctx, width, height } = canvasContext($("#scanCanvas"));
  ctx.clearRect(0, 0, width, height);
  const cx = width / 2, cy = height * 0.58, scale = Math.min(width * .43, height * .46) / frame.max_range;
  ctx.strokeStyle = "#26302d"; ctx.lineWidth = 1;
  for (const ratio of [.25, .5, .75, 1]) {
    ctx.beginPath(); ctx.arc(cx, cy, frame.max_range * scale * ratio, Math.PI * 1.25, Math.PI * 1.75); ctx.stroke();
    ctx.fillStyle = "#66736e"; ctx.font = "10px ui-monospace"; ctx.fillText(`${(frame.max_range * ratio).toFixed(0)}m`, cx + 5, cy - frame.max_range * scale * ratio + 12);
  }
  ctx.strokeStyle = "#3c4743"; ctx.beginPath(); ctx.moveTo(cx, cy + 12); ctx.lineTo(cx, 18); ctx.stroke();
  ctx.fillStyle = "#54e6d0";
  const span = frame.angle_max - frame.angle_min;
  frame.ranges.forEach((range, index) => {
    if (!Number.isFinite(range) || range <= 0 || range >= frame.max_range) return;
    const angle = frame.angle_min + span * index / Math.max(1, frame.ranges.length - 1);
    const x = cx - Math.sin(angle) * range * scale;
    const y = cy - Math.cos(angle) * range * scale;
    ctx.fillRect(x - 1.2, y - 1.2, 2.4, 2.4);
  });
  ctx.fillStyle = "#c9fa3f"; ctx.fillRect(cx - 7, cy - 11, 14, 22);
  ctx.strokeStyle = "#c9fa3f"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx, cy - 11); ctx.lineTo(cx, cy - 30); ctx.stroke();
}

function drawHistogram(values, edges) {
  const { context: ctx, width, height } = canvasContext($("#histogramCanvas"));
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 36, right: 12, top: 16, bottom: 28 }; const w = width - pad.left - pad.right, h = height - pad.top - pad.bottom;
  const max = Math.max(...values, 1); const barWidth = w / values.length;
  ctx.strokeStyle = "#35403d"; ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + h); ctx.lineTo(pad.left + w, pad.top + h); ctx.stroke();
  values.forEach((value, index) => {
    const barHeight = value / max * h;
    const center = (edges[index] + edges[index + 1]) / 2;
    ctx.fillStyle = Math.abs(center) <= .02 ? "#c9fa3f" : center < 0 ? "#54e6d0" : "#ff9d52";
    ctx.fillRect(pad.left + index * barWidth + 1, pad.top + h - barHeight, Math.max(1, barWidth - 2), barHeight);
  });
  ctx.fillStyle = "#82908a"; ctx.font = "10px ui-monospace"; ctx.textAlign = "center";
  ctx.fillText(edges[0].toFixed(2), pad.left, height - 7); ctx.fillText("0", pad.left + w / 2, height - 7); ctx.fillText(edges.at(-1).toFixed(2), pad.left + w, height - 7);
}

function startPlayback() {
  state.playing = true; $("#playButton").textContent = "Ⅱ";
  state.timer = setInterval(() => {
    let next = Number($("#frameSlider").value) + 1;
    if (next > Number($("#frameSlider").max)) next = 0;
    loadFrame(next);
  }, 100);
}
function stopPlayback() { state.playing = false; clearInterval(state.timer); state.timer = null; $("#playButton").textContent = "▶"; }

function renderRecordings() {
  const container = $("#recordingList");
  if (!state.overview.recordings.length) { container.innerHTML = '<div class="empty">Bag Managerの録画が見つかりません。</div>'; return; }
  container.innerHTML = state.overview.recordings.map((item) => `<div class="recording-row"><div><b>${escapeHtml(item.name)}</b><small>${escapeHtml(item.id)}</small></div><span>${fmtBytes(item.size_bytes)}</span><label><input type="checkbox" data-split="train" value="${escapeHtml(item.id)}" aria-label="Train ${escapeHtml(item.id)}"></label><label><input type="checkbox" data-split="val" value="${escapeHtml(item.id)}" aria-label="Val ${escapeHtml(item.id)}"></label></div>`).join("");
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
  $("#collectionWatchState").textContent = recordings.length ? `最新 ${fmtTime(recordings[0].modified_at)}` : "L1停止後に自動表示";
  $("#collectionRecent").innerHTML = recordings.length ? recordings.slice(0, 12).map((item) => {
    const annotation = item.annotation;
    const label = annotation ? `${collectionCategoryLabels[annotation.category] || annotation.category} · ${collectionOutcomeLabels[annotation.outcome] || annotation.outcome}` : "未分類";
    return `<button type="button" class="${item.id === state.collectionSelected ? "active" : ""}" data-collection-recording="${escapeHtml(item.id)}"><b>${escapeHtml(item.name)}</b><span class="collection-badge ${annotation ? "saved" : ""}">${escapeHtml(label)}</span><small>${escapeHtml(item.id)} · ${fmtTime(item.modified_at)}</small></button>`;
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

function renderCheckpoints() {
  const select = $("#pretrained"); const selected = select.value;
  select.innerHTML = '<option value="">使用しない</option>' + state.overview.checkpoints.filter((item) => item.best).map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)} · ${fmtTime(item.modified_at)}</option>`).join("");
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
}

function renderReadiness() {
  const train = state.overview.sequences.filter((item) => item.version === state.activeVersion && item.split === "train" && item.valid);
  const val = state.overview.sequences.filter((item) => item.version === state.activeVersion && item.split === "val" && item.valid);
  const trainSamples = train.reduce((total, item) => total + item.samples, 0);
  const valSamples = val.reduce((total, item) => total + item.samples, 0);
  $("#readiness").innerHTML = `<span>Dataset Version <b class="ok">${escapeHtml(state.activeVersion)}</b></span><span>Train dataset <b class="${train.length ? "ok" : "warn"}">${train.length ? `${fmtNumber(trainSamples)} samples` : "未準備"}</b></span><span>Validation dataset <b class="${val.length ? "ok" : "warn"}">${val.length ? `${fmtNumber(valSamples)} samples` : "未準備"}</b></span><span>出力 <b class="ok">Version別に自動作成</b></span>`;
  $("#trainingForm button[type=submit]").disabled = !train.length || !val.length;
}

async function submitPreprocess(event) {
  event.preventDefault();
  const selected = (split) => $$(`input[data-split="${split}"]:checked`).map((input) => input.value);
  const newVersion = $("#newVersion").value.trim();
  if (newVersion && !$("#newVersion").checkValidity()) { $("#newVersion").reportValidity(); return; }
  const datasetVersion = newVersion || $("#preprocessVersion").value;
  const payload = { dataset_version: datasetVersion, train: selected("train"), val: selected("val"), existing_policy: $("#existingPolicy").value, max_sync_delta: Number($("#syncDelta").value), max_range: Number($("#maxRange").value) };
  try { const job = await api("/api/jobs/preprocess", { method: "POST", body: JSON.stringify(payload) }); state.selectedJob = job.id; state.activeVersion = datasetVersion; localStorage.setItem("tinyLidarDatasetVersion", datasetVersion); $("#newVersion").value = ""; toast(`${datasetVersion} の前処理を開始しました`); showTab("jobs"); await refreshOverview(); }
  catch (error) { toast(error.message, true); }
}

async function submitTraining(event) {
  event.preventDefault();
  const payload = { dataset_version: state.activeVersion, architecture: $("#architecture").value, steering_only: $("#steeringOnly").checked, pretrained: $("#pretrained").value, device: $("#device").value, epochs: Number($("#epochs").value), batch_size: Number($("#batchSize").value), learning_rate: Number($("#learningRate").value), patience: Number($("#patience").value), workers: Number($("#workers").value) };
  try { const job = await api("/api/jobs/train", { method: "POST", body: JSON.stringify(payload) }); state.selectedJob = job.id; toast("学習ジョブを開始しました"); showTab("jobs"); await refreshOverview(); }
  catch (error) { toast(error.message, true); }
}

function renderJobs() {
  const jobs = state.overview.jobs;
  if (!jobs.length) { $("#jobList").innerHTML = '<div class="empty">まだジョブはありません。</div>'; return; }
  if (!state.selectedJob || !jobs.some((job) => job.id === state.selectedJob)) state.selectedJob = jobs[0].id;
  $("#jobList").innerHTML = jobs.map((job) => `<button class="job-item ${job.id === state.selectedJob ? "active" : ""}" data-job="${job.id}"><b>${job.kind === "train" ? "学習" : "前処理"}</b><span class="status ${job.status}">${job.status}</span><small>${job.id}${job.pid ? ` · PID ${job.pid}` : ""} · ${fmtTime(job.created_at)}</small></button>`).join("");
  $$(".job-item").forEach((item) => item.addEventListener("click", () => { state.selectedJob = item.dataset.job; renderJobs(); }));
  const job = jobs.find((item) => item.id === state.selectedJob);
  $("#logTitle").textContent = `${job.kind === "train" ? "学習" : "前処理"} · ${job.id} · ${job.status}`;
  $("#jobLog").textContent = job.log || "ジョブの開始を待っています…";
  $("#jobLog").scrollTop = $("#jobLog").scrollHeight;
  $("#cancelJob").hidden = !["queued", "running"].includes(job.status);
}

async function cancelSelectedJob() {
  if (!state.selectedJob) return;
  try { await api(`/api/jobs/${state.selectedJob}/cancel`, { method: "POST", body: "{}" }); toast("停止を要求しました"); await refreshOverview(); }
  catch (error) { toast(error.message, true); }
}

function applyPreset(name) {
  $$("[data-preset]").forEach((button) => button.classList.toggle("active", button.dataset.preset === name));
  if (name === "finetune") { $("#learningRate").value = "0.0001"; $("#epochs").value = "40"; $("#patience").value = "10"; if ($("#pretrained").options.length > 1) $("#pretrained").selectedIndex = 1; }
  else { $("#learningRate").value = "0.001"; $("#epochs").value = "100"; $("#patience").value = "15"; $("#pretrained").value = ""; }
}

function showTab(id) {
  $$(".tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === id));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === id));
  if (id === "explorer" && state.frame) setTimeout(() => { drawScan(state.frame); drawHistogram(state.detail.steering.histogram, state.detail.steering.edges); }, 0);
}

function bindEvents() {
  $$(".tabs button").forEach((button) => button.addEventListener("click", () => showTab(button.dataset.tab)));
  $("#datasetVersion").addEventListener("change", (event) => switchDatasetVersion(event.target.value));
  $("#sequenceSelect").addEventListener("change", loadSelectedSequence);
  $("#frameSlider").addEventListener("input", (event) => loadFrame(event.target.value));
  $("#frameNumber").addEventListener("change", (event) => loadFrame(event.target.value));
  $("#playButton").addEventListener("click", () => state.playing ? stopPlayback() : startPlayback());
  $("#preprocessForm").addEventListener("submit", submitPreprocess);
  $("#trainingForm").addEventListener("submit", submitTraining);
  $("#collectionForm").addEventListener("submit", saveCollectionAnnotation);
  $("#collectionRecording").addEventListener("change", (event) => selectCollectionRecording(event.target.value));
  $("#cancelJob").addEventListener("click", cancelSelectedJob);
  $("#refreshButton").addEventListener("click", () => refreshOverview());
  $$("[data-preset]").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
  window.addEventListener("resize", () => { if (state.frame) { drawScan(state.frame); drawHistogram(state.detail.steering.histogram, state.detail.steering.edges); } });
}

bindEvents();
refreshOverview(false);
setInterval(() => { if (state.overview?.jobs.some((job) => ["queued", "running"].includes(job.status))) refreshOverview(); }, 1500);
setInterval(pollLatestRecording, 2000);
