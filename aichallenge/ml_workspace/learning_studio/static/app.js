const state = {
  config: {},
  sequences: [],
  datasets: [],
  checkpoints: [],
  evaluations: [],
  assignments: new Map(),
  job: null,
  evaluation: null,
  frameIndex: 0,
  playing: false,
  playTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  return new Intl.NumberFormat("ja-JP").format(Number(value || 0));
}

function timestampName(prefix) {
  const now = new Date();
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
    "-",
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0"),
  ].join("");
  return `${prefix}_${stamp}`;
}

function formObject(form) {
  return Object.fromEntries(new FormData(form).entries());
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function toast(message, type = "success") {
  const item = document.createElement("div");
  item.className = `toast ${type === "error" ? "error" : ""}`;
  item.textContent = message;
  $("#toastRegion").append(item);
  setTimeout(() => item.remove(), 4200);
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Clipboard is not available");
}

function setView(name) {
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
}

async function refreshState({ quiet = false } = {}) {
  try {
    const payload = await api("/api/state");
    Object.assign(state, payload);
    renderAll();
    if (!quiet) toast("最新の状態を読み込みました。");
  } catch (error) {
    $("#connectionLabel").textContent = "Connection failed";
    toast(error.message, "error");
  }
}

function renderAll() {
  $("#connectionLabel").textContent = `Connected on ${state.config.python || "python"}`;
  $("#recordRoot").textContent = state.config.record_root || "/aichallenge/record";
  renderSequences();
  renderDatasetOptions();
  renderCheckpoints();
  renderEvaluationOptions();
  syncModelUi("extract");
  syncModelUi("train");
  syncModelUi("eval");
  renderJob(state.job);
}

function renderSequences() {
  const query = $("#sequenceFilter").value.trim().toLowerCase();
  const visible = state.sequences.filter((sequence) => {
    const text = [
      sequence.name,
      sequence.relative_path,
      ...sequence.topics.map((topic) => `${topic.name} ${topic.type}`),
    ].join(" ").toLowerCase();
    return !query || text.includes(query);
  });
  $("#bagCount").textContent = formatNumber(state.sequences.length);
  $("#messageCount").textContent = formatNumber(
    state.sequences.reduce((sum, sequence) => sum + sequence.messages, 0),
  );
  $("#sequenceEmpty").hidden = visible.length > 0;
  $("#sequenceRows").innerHTML = visible.map((sequence) => {
    const split = state.assignments.get(sequence.id) || "unused";
    const topics = sequence.topics.slice(0, 3)
      .map((topic) => `<span class="topic" title="${escapeHtml(topic.type)}">${escapeHtml(topic.name)}</span>`)
      .join("");
    const more = sequence.topics.length > 3
      ? `<span class="topic more-topics">+${sequence.topics.length - 3} more</span>`
      : "";
    return `
      <tr>
        <td>
          <select class="split-select" data-sequence-id="${sequence.id}" data-split="${split}">
            <option value="unused" ${split === "unused" ? "selected" : ""}>Unused</option>
            <option value="train" ${split === "train" ? "selected" : ""}>Train</option>
            <option value="val" ${split === "val" ? "selected" : ""}>Validation</option>
            <option value="both" ${split === "both" ? "selected" : ""}>Both</option>
          </select>
        </td>
        <td class="sequence-name">
          <strong title="${escapeHtml(sequence.relative_path)}">${escapeHtml(sequence.name)}</strong>
          <span>${escapeHtml(sequence.relative_path)}</span>
        </td>
        <td>${escapeHtml(sequence.duration)}</td>
        <td>${formatNumber(sequence.messages)}</td>
        <td><div class="topic-stack">${topics}${more}</div></td>
      </tr>
    `;
  }).join("");

  $$(".split-select").forEach((select) => {
    select.addEventListener("change", () => {
      state.assignments.set(select.dataset.sequenceId, select.value);
      select.dataset.split = select.value;
      renderSelectionMetrics();
    });
  });
  renderSelectionMetrics();
}

function renderSelectionMetrics() {
  let train = 0;
  let val = 0;
  state.sequences.forEach((sequence) => {
    const split = state.assignments.get(sequence.id) || "unused";
    if (split === "train" || split === "both") train += 1;
    if (split === "val" || split === "both") val += 1;
  });
  $("#trainCount").textContent = formatNumber(train);
  $("#valCount").textContent = formatNumber(val);
  $("#selectionSummary").textContent = train || val
    ? `Train ${train} sequence / Validation ${val} sequence を抽出します。`
    : "sequence を選択してください";
}

function optionMarkup(items, label, value, emptyText) {
  if (!items.length) return `<option value="">${escapeHtml(emptyText)}</option>`;
  return items.map((item) => `<option value="${escapeHtml(value(item))}">${escapeHtml(label(item))}</option>`).join("");
}

function modelItems(items, modelType) {
  return items.filter((item) => item.model_type === modelType);
}

function renderDatasetSelect(select, modelType) {
  const current = select.value;
  const datasets = modelItems(state.datasets, modelType);
  select.innerHTML = optionMarkup(
    datasets,
    (dataset) => `${dataset.name} · train ${formatNumber(dataset.train_samples)} / val ${formatNumber(dataset.val_samples)}`,
    (dataset) => dataset.name,
    "Dataset がありません",
  );
  if ([...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
}

function renderDatasetOptions() {
  renderDatasetSelect($("#trainDataset"), $("#trainModel").value);
  renderDatasetSelect($("#evalDataset"), $("#evalModel").value);
  syncDatasetShape("train");
}

function renderCheckpoints() {
  const trainModel = $("#trainModel").value;
  const evalModel = $("#evalModel").value;
  const trainCheckpoints = modelItems(state.checkpoints, trainModel);
  const evalCheckpoints = modelItems(state.checkpoints, evalModel);
  $("#checkpointCount").textContent = formatNumber(trainCheckpoints.length);
  $("#checkpointList").innerHTML = trainCheckpoints.length
    ? trainCheckpoints.map((checkpoint) => `
        <div class="artifact">
          <strong>${escapeHtml(checkpoint.name)} <small>${escapeHtml(checkpoint.model_label)}</small></strong>
          <span title="${escapeHtml(checkpoint.path)}">${escapeHtml(checkpoint.path)}</span>
        </div>
      `).join("")
    : '<div class="empty-inline">Checkpoint はまだありません。</div>';

  const evalMarkup = optionMarkup(
    evalCheckpoints,
    (checkpoint) => checkpoint.name,
    (checkpoint) => checkpoint.path,
    "Checkpoint がありません",
  );
  const previousEval = $("#evalCheckpoint").value;
  $("#evalCheckpoint").innerHTML = evalMarkup;
  if ([...$("#evalCheckpoint").options].some((option) => option.value === previousEval)) {
    $("#evalCheckpoint").value = previousEval;
  }
  syncEvaluationModelConfig();

  const previousPretrained = $("#pretrainedCheckpoint").value;
  const pretrainedMarkup = optionMarkup(
    trainCheckpoints,
    (checkpoint) => checkpoint.name,
    (checkpoint) => checkpoint.path,
    "Checkpoint がありません",
  );
  $("#pretrainedCheckpoint").innerHTML = `<option value="">None</option>${pretrainedMarkup}`;
  if ([...$("#pretrainedCheckpoint").options].some((option) => option.value === previousPretrained)) {
    $("#pretrainedCheckpoint").value = previousPretrained;
  }
}

function syncEvaluationModelConfig() {
  const checkpoint = state.checkpoints.find((item) => item.path === $("#evalCheckpoint").value);
  if (!checkpoint?.model) return;
  const form = $("#evaluationForm");
  form.elements.image_width.value = checkpoint.model.image_width ?? 200;
  form.elements.image_height.value = checkpoint.model.image_height ?? 66;
  form.elements.output_dim.value = checkpoint.model.output_dim ?? 2;
  form.elements.color_space.value = checkpoint.model.color_space ?? "yuv";
  form.elements.architecture.value = checkpoint.model.architecture ?? "TinyLidarNet";
  form.elements.input_dim.value = checkpoint.model.input_dim ?? 750;
  form.elements.max_range.value = checkpoint.model.max_range ?? 30;
}

function renderEvaluationOptions() {
  const select = $("#existingEvaluation");
  const current = select.value;
  const evaluations = modelItems(state.evaluations, $("#evalModel").value);
  select.innerHTML = optionMarkup(
    evaluations,
    (evaluation) => `${evaluation.name} · ${evaluation.split} · ${Number(evaluation.mean_mae || 0).toFixed(4)}`,
    (evaluation) => evaluation.id,
    "Evaluation がありません",
  );
  if ([...select.options].some((option) => option.value === current)) select.value = current;
}

function syncDatasetShape(context) {
  const modelSelect = $(`#${context}Model`);
  const datasetSelect = $(`#${context}Dataset`);
  if (!modelSelect || !datasetSelect || modelSelect.value !== "tiny_lidar_net") return;
  const dataset = state.datasets.find(
    (item) => item.model_type === "tiny_lidar_net" && item.name === datasetSelect.value,
  );
  const inputDim = dataset?.input_shape?.[0];
  if (inputDim && context === "train") {
    $("#trainForm").elements.input_dim.value = inputDim;
  }
}

function syncModelUi(context) {
  const modelType = $(`#${context}Model`).value;
  const form = context === "extract"
    ? $("#extractForm")
    : context === "train"
      ? $("#trainForm")
      : $("#evaluationForm");
  $$("[data-model-only]", form).forEach((group) => {
    const active = group.dataset.modelOnly === modelType;
    group.hidden = !active;
    $$("input, select", group).forEach((control) => {
      control.disabled = !active;
    });
  });
  if (context === "train") {
    const tiny = modelType === "tiny_lidar_net";
    $("#trainModelTitle").textContent = tiny ? "TinyLiDARNet" : "PilotNet";
    $("#trainModelChip").textContent = tiny ? "TinyLiDARNet / 2D LiDAR" : "PilotNet / Camera";
  }
  if (context === "eval") syncGradCamUi(modelType);
}

function syncGradCamUi(modelType) {
  const supported = modelType === "pilot_net";
  const toggle = $("#gradCamToggle");
  toggle.disabled = !supported;
  if (!supported) toggle.checked = false;
  $("#gradCamControl").classList.toggle("disabled", !supported);
  $("#gradCamControl").title = supported
    ? "PilotNet steering output / conv5"
    : "Grad-CAM is currently available for PilotNet only";
}

function renderJob(job) {
  state.job = job;
  const running = job && ["queued", "running"].includes(job.status);
  $("#jobTitle").textContent = job ? `${job.kind.toUpperCase()} · ${job.name}` : "No active job";
  $("#jobDetail").textContent = job
    ? `${job.status}${job.error ? ` · ${job.error}` : ""}`
    : "Idle";
  $("#jobDot").className = `status-dot ${job?.status || ""}`;
  $("#jobProgress").style.width = `${Math.max(0, Math.min(1, job?.progress || 0)) * 100}%`;
  $("#stopJobButton").disabled = !running;
  $("#consoleTitle").textContent = job ? `${job.kind} / ${job.name}` : "Console";
  $("#consoleOutput").textContent = job?.log?.length ? job.log.join("\n") : "No job output.";
  const console = $("#consoleOutput");
  if ($("#consoleDialog").open) console.scrollTop = console.scrollHeight;
}

async function pollJob() {
  try {
    const payload = await api("/api/job");
    const previousStatus = state.job?.status;
    renderJob(payload.job);
    if (
      payload.job
      && previousStatus
      && previousStatus !== payload.job.status
      && ["succeeded", "failed", "cancelled"].includes(payload.job.status)
    ) {
      toast(
        payload.job.status === "succeeded"
          ? `${payload.job.name} が完了しました。`
          : `${payload.job.name}: ${payload.job.status}`,
        payload.job.status === "succeeded" ? "success" : "error",
      );
      await refreshState({ quiet: true });
    }
  } catch {
    // The full refresh path reports connection failures.
  }
}

async function postForm(path, payload) {
  try {
    const response = await api(path, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderJob(response.job);
    toast(`${response.job.name} を開始しました。`);
    $("#consoleDialog").showModal();
  } catch (error) {
    toast(error.message, "error");
  }
}

function numericPayload(payload, names) {
  names.forEach((name) => {
    payload[name] = Number(payload[name]);
  });
  return payload;
}

async function loadEvaluation(name) {
  if (!name) {
    toast("読み込む evaluation を選択してください。", "error");
    return;
  }
  stopPlayback();
  try {
    const detail = await api(`/api/evaluations/${encodeURIComponent(name)}`);
    state.evaluation = detail;
    state.frameIndex = 0;
    syncGradCamUi(detail.model_type);
    $("#frameSlider").max = Math.max(0, detail.summary.frame_count - 1);
    $("#frameSlider").value = "0";
    renderEvaluationSummary();
    drawErrorChart();
    await showFrame(0);
    setView("evaluate");
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderEvaluationSummary() {
  const evaluation = state.evaluation;
  $("#evaluationSummary").innerHTML = [
    ["Frames", formatNumber(evaluation.summary.frame_count)],
    ["Mean MAE", Number(evaluation.summary.mean_mae).toFixed(4)],
    ["P95", Number(evaluation.summary.p95_mae).toFixed(4)],
    ["Max", Number(evaluation.summary.max_mae).toFixed(4)],
  ].map(([label, value]) => `
    <span class="summary-pill">${label}<strong>${value}</strong></span>
  `).join("");
  $("#worstFrames").innerHTML = evaluation.worst.map((frame) => `
    <button class="worst-frame" data-frame-index="${frame.index}">
      <strong>MAE ${Number(frame.mae).toFixed(4)}</strong>
      <span>Frame ${formatNumber(frame.index + 1)}</span>
      <span title="${escapeHtml(frame.sequence)}">${escapeHtml(frame.sequence)}</span>
    </button>
  `).join("");
  $$(".worst-frame").forEach((button) => {
    button.addEventListener("click", () => showFrame(Number(button.dataset.frameIndex)));
  });
}

async function showFrame(index) {
  const evaluation = state.evaluation;
  if (!evaluation?.summary.frame_count) return;
  const clamped = Math.max(0, Math.min(evaluation.summary.frame_count - 1, Number(index)));
  state.frameIndex = clamped;
  $("#frameSlider").value = String(clamped);
  const overlay = $("#overlayToggle").checked ? 1 : 0;
  const gradCam = $("#gradCamToggle").checked ? 1 : 0;
  $("#evaluationFrame").src = `/api/evaluations/${encodeURIComponent(evaluation.id)}/frame.jpg?index=${clamped}&overlay=${overlay}&gradcam=${gradCam}&t=${Date.now()}`;
  $("#evaluationFrame").hidden = false;
  $("#viewerPlaceholder").hidden = true;
  try {
    const info = await api(
      `/api/evaluations/${encodeURIComponent(evaluation.id)}/frame-info?index=${clamped}`,
    );
    if (state.frameIndex !== clamped) return;
    $("#frameSequence").textContent = info.sequence;
    $("#frameNumber").textContent = `Frame ${formatNumber(clamped + 1)} / ${formatNumber(evaluation.summary.frame_count)}`;
    $("#frameMae").textContent = `MAE ${Number(info.mae).toFixed(4)}`;
    $("#steerTarget").textContent = Number(info.target.steer).toFixed(4);
    $("#steerPrediction").textContent = Number(info.prediction.steer).toFixed(4);
    $("#accelTarget").textContent = info.target.acceleration == null
      ? "—"
      : Number(info.target.acceleration).toFixed(4);
    $("#accelPrediction").textContent = info.prediction.acceleration == null
      ? "—"
      : Number(info.prediction.acceleration).toFixed(4);
    drawErrorChart();
  } catch (error) {
    stopPlayback();
    toast(error.message, "error");
  }
}

function drawErrorChart() {
  const canvas = $("#errorChart");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0b1117";
  context.fillRect(0, 0, width, height);
  const series = state.evaluation?.series;
  if (!series?.mae?.length) {
    context.fillStyle = "#667485";
    context.font = "13px system-ui";
    context.fillText("No evaluation loaded", 18, 30);
    return;
  }
  const maximum = Math.max(...series.mae, 0.0001);
  context.strokeStyle = "#25323e";
  context.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) {
    const y = (height / 4) * row;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  const gradient = context.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#6dd3ad");
  gradient.addColorStop(1, "#ef7b82");
  context.strokeStyle = gradient;
  context.lineWidth = 2;
  context.beginPath();
  series.mae.forEach((value, index) => {
    const x = (index / Math.max(1, series.mae.length - 1)) * width;
    const y = height - (value / maximum) * (height - 12) - 6;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
  const currentRatio = state.frameIndex / Math.max(1, state.evaluation.summary.frame_count - 1);
  const currentX = currentRatio * width;
  context.strokeStyle = "#f1f4f4";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(currentX, 0);
  context.lineTo(currentX, height);
  context.stroke();
}

function stopPlayback() {
  state.playing = false;
  clearInterval(state.playTimer);
  state.playTimer = null;
  $("#playButton").textContent = "▶";
  $("#playButton").title = "再生";
}

function togglePlayback() {
  if (!state.evaluation) return;
  if (state.playing) {
    stopPlayback();
    return;
  }
  state.playing = true;
  $("#playButton").textContent = "Ⅱ";
  $("#playButton").title = "停止";
  const requestedFps = Number($("#playbackSpeed").value);
  const fps = $("#gradCamToggle").checked ? Math.min(2, requestedFps) : requestedFps;
  state.playTimer = setInterval(() => {
    const next = state.frameIndex + 1;
    if (next >= state.evaluation.summary.frame_count) {
      stopPlayback();
      return;
    }
    showFrame(next);
  }, 1000 / fps);
}

function bindEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));
  $("#refreshButton").addEventListener("click", () => refreshState());
  $("#sequenceFilter").addEventListener("input", renderSequences);
  $("#extractModel").addEventListener("change", () => {
    syncModelUi("extract");
    const prefix = $("#extractModel").value === "tiny_lidar_net"
      ? "tiny_lidar_dataset"
      : "pilotnet_dataset";
    $("#datasetName").value = timestampName(prefix);
  });
  $("#trainModel").addEventListener("change", () => {
    syncModelUi("train");
    renderDatasetSelect($("#trainDataset"), $("#trainModel").value);
    syncDatasetShape("train");
    renderCheckpoints();
    const prefix = $("#trainModel").value === "tiny_lidar_net"
      ? "tinylidar"
      : "pilotnet";
    $("#runName").value = timestampName(prefix);
  });
  $("#evalModel").addEventListener("change", () => {
    syncModelUi("eval");
    renderDatasetSelect($("#evalDataset"), $("#evalModel").value);
    renderCheckpoints();
    renderEvaluationOptions();
  });
  $("#trainDataset").addEventListener("change", () => syncDatasetShape("train"));

  $("#extractForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const payload = formObject(event.currentTarget);
    const numericFields = payload.model_type === "pilot_net"
      ? ["image_width", "image_height", "crop_top_ratio", "workers"]
      : ["max_range", "workers"];
    numericPayload(payload, numericFields);
    payload.assignments = state.sequences.map((sequence) => ({
      id: sequence.id,
      split: state.assignments.get(sequence.id) || "unused",
    }));
    postForm("/api/extract", payload);
  });

  $("#trainForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const payload = formObject(event.currentTarget);
    const numericFields = [
      "epochs",
      "batch_size",
      "num_workers",
      "early_stop_patience",
      "lr",
      "steer_weight",
      "accel_weight",
    ];
    if (payload.model_type === "pilot_net") {
      numericFields.push(
        "image_height",
        "image_width",
        "output_dim",
        "weight_decay",
        "shift_range",
        "steer_correction_per_pixel",
      );
    } else {
      numericFields.push("input_dim", "max_range");
    }
    numericPayload(payload, numericFields);
    postForm("/api/train", payload);
  });

  $("#evaluationForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const payload = formObject(event.currentTarget);
    const numericFields = ["output_dim", "batch_size"];
    if (payload.model_type === "pilot_net") {
      numericFields.push("image_width", "image_height");
    } else {
      numericFields.push("input_dim", "max_range");
    }
    numericPayload(payload, numericFields);
    postForm("/api/evaluate", payload);
  });

  $("#loadEvaluationButton").addEventListener("click", () => loadEvaluation($("#existingEvaluation").value));
  $("#evalCheckpoint").addEventListener("change", syncEvaluationModelConfig);
  $("#previousButton").addEventListener("click", () => showFrame(state.frameIndex - 1));
  $("#nextButton").addEventListener("click", () => showFrame(state.frameIndex + 1));
  $("#playButton").addEventListener("click", togglePlayback);
  $("#frameSlider").addEventListener("input", (event) => showFrame(Number(event.target.value)));
  $("#overlayToggle").addEventListener("change", () => showFrame(state.frameIndex));
  $("#gradCamToggle").addEventListener("change", () => {
    if (state.playing) {
      stopPlayback();
      togglePlayback();
    }
    showFrame(state.frameIndex);
  });
  $("#playbackSpeed").addEventListener("change", () => {
    if (state.playing) {
      stopPlayback();
      togglePlayback();
    }
  });

  $("#consoleButton").addEventListener("click", () => $("#consoleDialog").showModal());
  $("#closeConsoleButton").addEventListener("click", () => $("#consoleDialog").close());
  $("#copyConsoleButton").addEventListener("click", async () => {
    const output = $("#consoleOutput").textContent || "";
    try {
      await copyText(output);
      const button = $("#copyConsoleButton");
      button.textContent = "Copied";
      toast("Console log をコピーしました。");
      setTimeout(() => {
        button.textContent = "Copy log";
      }, 1600);
    } catch (error) {
      toast(`コピーできませんでした: ${error.message}`, "error");
    }
  });
  $("#stopJobButton").addEventListener("click", async () => {
    try {
      const payload = await api("/api/job/cancel", { method: "POST", body: "{}" });
      renderJob(payload.job);
      toast("停止を要求しました。");
    } catch (error) {
      toast(error.message, "error");
    }
  });
}

function initializeDefaults() {
  $("#datasetName").value = timestampName("pilotnet_dataset");
  $("#runName").value = timestampName("pilotnet");
  $("#evaluationName").value = timestampName("validation_review");
  syncModelUi("extract");
  syncModelUi("train");
  syncModelUi("eval");
}

initializeDefaults();
bindEvents();
refreshState({ quiet: true });
setInterval(pollJob, 1000);
