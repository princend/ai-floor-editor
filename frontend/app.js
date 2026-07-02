const fileInput = document.getElementById("fileInput");
const canvas = document.getElementById("imageCanvas");
const emptyState = document.getElementById("emptyState");
const statusBox = document.getElementById("status");
const originalViewButton = document.getElementById("originalViewButton");
const maskViewButton = document.getElementById("maskViewButton");
const previewViewButton = document.getElementById("previewViewButton");
const downloadButton = document.getElementById("downloadButton");
const maskButton = document.getElementById("maskButton");
const undoPointButton = document.getElementById("undoPointButton");
const clearPointsButton = document.getElementById("clearPointsButton");
const includeModeButton = document.getElementById("includeModeButton");
const excludeModeButton = document.getElementById("excludeModeButton");
const inpaintButton = document.getElementById("inpaintButton");
const materialGrid = document.getElementById("materialGrid");
const promptInput = document.getElementById("promptInput");
const proceduralEngineButton = document.getElementById("proceduralEngineButton");
const textureEngineButton = document.getElementById("textureEngineButton");
const diffusionEngineButton = document.getElementById("diffusionEngineButton");
const diffusionControls = document.getElementById("diffusionControls");
const textureControls = document.getElementById("textureControls");
const textureInput = document.getElementById("textureInput");
const texturePreview = document.getElementById("texturePreview");
const textureScaleInput = document.getElementById("textureScaleInput");
const textureScaleValue = document.getElementById("textureScaleValue");
const strengthInput = document.getElementById("strengthInput");
const strengthValue = document.getElementById("strengthValue");
const stepsInput = document.getElementById("stepsInput");
const stepsValue = document.getElementById("stepsValue");
const guidanceInput = document.getElementById("guidanceInput");
const guidanceValue = document.getElementById("guidanceValue");
const previewStatus = document.getElementById("previewStatus");
const apiResult = document.getElementById("apiResult");
const zhButton = document.getElementById("zhButton");
const enButton = document.getElementById("enButton");

const DEFAULT_NEGATIVE_PROMPT =
  "text, letters, words, logo, watermark, signature, caption, label, typography, numbers, people, warped geometry, distorted furniture, distorted walls, blurry, low quality";

const translations = {
  zh: {
    appTitle: "AI 地板更換 MVP",
    appSubtitle: "上傳室內照片、點選地板，預覽新的地板材質。",
    uploadImage: "上傳圖片",
    viewOriginal: "原圖",
    viewMask: "Mask",
    viewPreview: "預覽",
    download: "下載",
    emptyState: "上傳室內照片開始。",
    maskTitle: "地板 Mask",
    noImage: "尚未載入圖片。",
    includeFloor: "包含地板",
    excludeObject: "排除物件",
    generateMask: "由點位產生 Mask",
    undoPoint: "復原上一點",
    clearPoints: "清除點位",
    floorPreviewTitle: "地板預覽",
    engine: "生成方式",
    fastPreview: "快速預覽",
    textureMatch: "材質貼圖",
    aiInpaint: "AI 重繪",
    uploadTexture: "上傳材質圖",
    noTexture: "尚未載入材質圖。",
    textureScale: "材質比例",
    material: "材質",
    oak: "橡木",
    walnut: "胡桃木",
    tile: "磁磚",
    concrete: "水泥",
    prompt: "補充描述",
    promptPlaceholder: "例如：顏色偏暖、木紋更明顯。主要材質由上方按鈕決定。",
    strength: "替換強度",
    aiSteps: "AI 步數",
    promptGuidance: "Prompt 權重",
    generateFloorPreview: "產生地板預覽",
    generateTexturePreview: "產生材質預覽",
    generateAiInpaint: "產生 AI 重繪",
    ready: "就緒。",
    uploadingImage: "正在上傳圖片...",
    imageLoaded: "圖片已載入。請點選一個或多個地板點位。",
    pointsCleared: "點位已清除。請點選一個或多個地板點位。",
    allPointsRemoved: "所有點位已移除。",
    includeMode: "包含模式：點選地板區域。",
    excludeMode: "排除模式：點選椅子、沙發或不應該進入 mask 的物件。",
    generatingMask: "正在產生 mask...",
    maskRunning: "Mask 產生中... {seconds}s",
    addIncludePoint: "產生 mask 前至少要有一個包含地板點。",
    maskReady: "Mask 完成：{include} 個包含點、{exclude} 個排除點。覆蓋率：{coverage}%。後端耗時：{backendSeconds}s；總等待：{clientSeconds}s。演算法：{algorithm}.{device}",
    needMask: "請先產生地板 mask，再建立預覽。",
    needTexture: "使用材質貼圖前，請先上傳材質圖片。",
    uploadingTexture: "正在上傳材質圖...",
    textureLoaded: "材質已載入：{width}x{height}。",
    textureReady: "材質貼圖已就緒。",
    texturePrompt: "請上傳材質圖。",
    aiMode: "AI 重繪會在本機使用 Stable Diffusion，第一次可能較久。",
    aiPrompt: "AI 重繪會在本機載入模型。第一次可能需要下載與載入模型。",
    textureMode: "材質貼圖會使用你上傳的材質圖片。",
    fastMode: "快速預覽使用本機程式產生材質。",
    fastReady: "快速預覽已就緒。",
    aiBusy: "AI 重繪執行中。第一次可能需要數分鐘下載與載入模型。",
    textureBusy: "正在把上傳材質套用到地板 mask。",
    fastBusy: "正在產生快速地板預覽。",
    elapsed: "已用時：{seconds} 秒。",
    aiRunning: "AI 重繪中... {seconds}s",
    generating: "產生中... {seconds}s",
    previewReady: "{engine} 完成。{material}{device}",
    materialLabel: "材質：{material}。",
    deviceLabel: "裝置：{device}。",
    floorPreviewLabel: "地板預覽",
    texturePreviewLabel: "材質預覽",
    aiInpaintLabel: "AI 重繪",
    pointSummary: "已選取 {include} 個包含點、{exclude} 個排除點。",
    imageLoadFailed: "無法載入圖片。",
    requestFailed: "請求失敗：{status}。",
  },
  en: {
    appTitle: "AI Floor Editor MVP",
    appSubtitle: "Upload an interior photo, mark the floor, and preview a replacement material.",
    uploadImage: "Upload image",
    viewOriginal: "Original",
    viewMask: "Mask",
    viewPreview: "Preview",
    download: "Download",
    emptyState: "Upload a room photo to start.",
    maskTitle: "Mask",
    noImage: "No image loaded.",
    includeFloor: "Include floor",
    excludeObject: "Exclude object",
    generateMask: "Generate from points",
    undoPoint: "Undo last point",
    clearPoints: "Clear points",
    floorPreviewTitle: "Floor Preview",
    engine: "Engine",
    fastPreview: "Fast preview",
    textureMatch: "Texture match",
    aiInpaint: "AI inpaint",
    uploadTexture: "Upload texture",
    noTexture: "No texture loaded.",
    textureScale: "Texture scale",
    material: "Material",
    oak: "Oak",
    walnut: "Walnut",
    tile: "Tile",
    concrete: "Concrete",
    prompt: "Extra notes",
    promptPlaceholder: "Example: warmer color, more visible grain. The selected material controls the main result.",
    strength: "Strength",
    aiSteps: "AI steps",
    promptGuidance: "Prompt guidance",
    generateFloorPreview: "Generate floor preview",
    generateTexturePreview: "Generate texture preview",
    generateAiInpaint: "Generate AI inpaint",
    ready: "Ready.",
    uploadingImage: "Uploading image...",
    imageLoaded: "Image loaded. Click one or more floor points.",
    pointsCleared: "Points cleared. Click one or more floor points.",
    allPointsRemoved: "All points removed.",
    includeMode: "Include mode: click floor areas.",
    excludeMode: "Exclude mode: click chairs, sofas, or objects to remove.",
    generatingMask: "Generating mask...",
    maskRunning: "Mask running... {seconds}s",
    addIncludePoint: "Add at least one include floor point before generating a mask.",
    maskReady: "Mask ready from {include} include and {exclude} exclude point(s). Coverage: {coverage}%. Backend: {backendSeconds}s; total wait: {clientSeconds}s. Algorithm: {algorithm}.{device}",
    needMask: "Generate a mask before creating a floor preview.",
    needTexture: "Upload a material texture before using texture match.",
    uploadingTexture: "Uploading texture...",
    textureLoaded: "Texture loaded: {width}x{height}.",
    textureReady: "Texture match is ready.",
    texturePrompt: "Upload a material texture.",
    aiMode: "AI inpaint uses Stable Diffusion locally. First run can take a while.",
    aiPrompt: "AI inpaint is local. First run downloads and loads the model.",
    textureMode: "Texture match uses your uploaded material image.",
    fastMode: "Fast preview uses the procedural local renderer.",
    fastReady: "Fast preview is ready.",
    aiBusy: "AI inpaint is running. First run can spend several minutes downloading and loading the model.",
    textureBusy: "Applying uploaded texture to the floor mask.",
    fastBusy: "Generating fast floor preview.",
    elapsed: "Elapsed: {seconds}s.",
    aiRunning: "AI inpaint running... {seconds}s",
    generating: "Generating... {seconds}s",
    previewReady: "{engine} ready. {material}{device}",
    materialLabel: "Material: {material}.",
    deviceLabel: "Device: {device}.",
    floorPreviewLabel: "floor preview",
    texturePreviewLabel: "texture preview",
    aiInpaintLabel: "AI inpaint",
    pointSummary: "{include} include and {exclude} exclude point(s) selected.",
    imageLoadFailed: "Could not load image.",
    requestFailed: "Request failed with {status}.",
  },
};

const ctx = canvas.getContext("2d");
const state = {
  image: null,
  imageMeta: null,
  points: [],
  pointMode: "include",
  viewMode: "original",
  overlay: null,
  result: null,
  overlayUrl: null,
  resultUrl: null,
  hasMask: false,
  engine: "procedural",
  lang: "zh",
  materialMeta: null,
  materialKey: "oak",
  previewTimer: null,
  maskTimer: null,
};

applyLanguage("zh");

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  setStatus(t("uploadingImage"));
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/images", {
      method: "POST",
      body: formData,
    });
    const data = await readJson(response);
    state.imageMeta = data;
    state.points = [];
    state.viewMode = "original";
    state.overlay = null;
    state.result = null;
    state.overlayUrl = null;
    state.resultUrl = null;
    state.hasMask = false;
    await loadImage(data.url);
    maskButton.disabled = true;
    undoPointButton.disabled = true;
    clearPointsButton.disabled = true;
    inpaintButton.disabled = false;
    setViewControls();
    apiResult.textContent = "";
    setStatus(t("imageLoaded"));
  } catch (error) {
    setStatus(error.message, true);
  }
});

canvas.addEventListener("click", async (event) => {
  if (!state.image || !state.imageMeta) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.round((event.clientX - rect.left) * scaleX);
  const y = Math.round((event.clientY - rect.top) * scaleY);

  state.points.push({ x, y, label: state.pointMode === "include" ? 1 : 0 });
  state.overlay = null;
  state.result = null;
  state.overlayUrl = null;
  state.resultUrl = null;
  state.hasMask = false;
  state.viewMode = "original";
  maskButton.disabled = false;
  undoPointButton.disabled = false;
  clearPointsButton.disabled = false;
  setViewControls();
  draw();
  setStatus(pointSummary());
});

maskButton.addEventListener("click", async () => {
  if (!state.imageMeta || state.points.length === 0) return;
  await generateMask();
});

clearPointsButton.addEventListener("click", () => {
  state.points = [];
  state.overlay = null;
  state.result = null;
  state.overlayUrl = null;
  state.resultUrl = null;
  state.hasMask = false;
  state.viewMode = "original";
  maskButton.disabled = true;
  undoPointButton.disabled = true;
  clearPointsButton.disabled = true;
  setViewControls();
  draw();
  setStatus(t("pointsCleared"));
});

undoPointButton.addEventListener("click", () => {
  if (state.points.length === 0) return;
  state.points.pop();
  state.overlay = null;
  state.result = null;
  state.overlayUrl = null;
  state.resultUrl = null;
  state.hasMask = false;
  state.viewMode = "original";
  maskButton.disabled = state.points.length === 0;
  undoPointButton.disabled = state.points.length === 0;
  clearPointsButton.disabled = state.points.length === 0;
  setViewControls();
  draw();
  setStatus(state.points.length === 0 ? t("allPointsRemoved") : pointSummary());
});

zhButton.addEventListener("click", () => applyLanguage("zh"));
enButton.addEventListener("click", () => applyLanguage("en"));
includeModeButton.addEventListener("click", () => setPointMode("include"));
excludeModeButton.addEventListener("click", () => setPointMode("exclude"));
proceduralEngineButton.addEventListener("click", () => setEngine("procedural"));
textureEngineButton.addEventListener("click", () => setEngine("texture"));
diffusionEngineButton.addEventListener("click", () => setEngine("diffusion"));
originalViewButton.addEventListener("click", () => setViewMode("original"));
maskViewButton.addEventListener("click", () => setViewMode("mask"));
previewViewButton.addEventListener("click", () => setViewMode("preview"));
downloadButton.addEventListener("click", downloadCurrentView);
materialGrid.addEventListener("click", (event) => {
  const button = event.target.closest(".material-button");
  if (!button) return;

  state.materialKey = button.dataset.materialKey || "oak";
  for (const materialButton of materialGrid.querySelectorAll(".material-button")) {
    materialButton.classList.toggle("active", materialButton === button);
  }
});
strengthInput.addEventListener("input", () => {
  strengthValue.value = Number(strengthInput.value).toFixed(2);
});
textureScaleInput.addEventListener("input", () => {
  textureScaleValue.value = Number(textureScaleInput.value).toFixed(2);
});
stepsInput.addEventListener("input", () => {
  stepsValue.value = stepsInput.value;
});
guidanceInput.addEventListener("input", () => {
  guidanceValue.value = Number(guidanceInput.value).toFixed(1);
});
textureInput.addEventListener("change", async () => {
  const file = textureInput.files?.[0];
  if (!file) return;

  setPreviewStatus(t("uploadingTexture"));
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/materials", {
      method: "POST",
      body: formData,
    });
    const data = await readJson(response);
    state.materialMeta = data;
    texturePreview.innerHTML = "";
    const image = document.createElement("img");
    image.src = `${data.url}?t=${Date.now()}`;
    image.alt = t("uploadTexture");
    texturePreview.appendChild(image);
    setPreviewStatus(t("textureLoaded", { width: data.width, height: data.height }));
  } catch (error) {
    setPreviewStatus(error.message, true);
  }
});

inpaintButton.addEventListener("click", async () => {
  if (!state.imageMeta) return;

  if (!state.hasMask) {
    setStatus(t("needMask"), true);
    return;
  }

  if (state.engine === "texture" && !state.materialMeta) {
    setStatus(t("needTexture"), true);
    setPreviewStatus(t("needTexture"), true);
    return;
  }

  const isDiffusion = state.engine === "diffusion";
  startPreviewBusy(state.engine);
  inpaintButton.disabled = true;
  try {
    const response = await fetch(`/api/images/${state.imageMeta.image_id}/inpaint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: state.engine,
        prompt: promptInput.value,
        material_key: state.materialKey,
        negative_prompt: DEFAULT_NEGATIVE_PROMPT,
        strength: Number(strengthInput.value),
        steps: Number(stepsInput.value),
        guidance_scale: Number(guidanceInput.value),
        material_id: state.materialMeta?.material_id || null,
        texture_scale: Number(textureScaleInput.value),
      }),
    });
    const data = await readJson(response);
    apiResult.textContent = JSON.stringify(data, null, 2);
    state.resultUrl = data.output_url;
    state.result = await imageFromUrl(`${data.output_url}?t=${Date.now()}`);
    state.viewMode = "preview";
    setViewControls();
    draw();
    const engineLabel = labelForAlgorithm(data.stats.algorithm);
    const material = data.stats.material ? t("materialLabel", { material: data.stats.material }) : "";
    const device = data.stats.device ? t("deviceLabel", { device: data.stats.device }) : "";
    const message = t("previewReady", { engine: engineLabel, material, device });
    setStatus(message);
    setPreviewStatus(message);
  } catch (error) {
    setStatus(error.message, true);
    setPreviewStatus(error.message, true);
  } finally {
    stopPreviewBusy();
    inpaintButton.disabled = false;
    updateInpaintButtonText();
  }
});

async function generateMask() {
  const startedAt = Date.now();
  startMaskBusy(startedAt);
  maskButton.disabled = true;

  try {
    if (!state.points.some((point) => point.label === 1)) {
      throw new Error(t("addIncludePoint"));
    }

    const response = await fetch(`/api/images/${state.imageMeta.image_id}/mask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: state.points }),
    });
    const data = await readJson(response);
    const clientSeconds = ((Date.now() - startedAt) / 1000).toFixed(2);
    state.overlayUrl = data.overlay_url;
    state.overlay = await imageFromUrl(`${data.overlay_url}?t=${Date.now()}`);
    state.result = null;
    state.resultUrl = null;
    state.hasMask = true;
    state.viewMode = "mask";
    setViewControls();
    draw();
    const device = data.stats.device ? ` Device: ${data.stats.device}.` : "";
    setStatus(t("maskReady", {
      include: data.stats.positive_point_count || 0,
      exclude: data.stats.negative_point_count || 0,
      coverage: (data.stats.mask_coverage * 100).toFixed(1),
      backendSeconds: Number(data.stats.elapsed_seconds || 0).toFixed(2),
      clientSeconds,
      algorithm: data.stats.algorithm,
      device: data.stats.device ? ` ${t("deviceLabel", { device: data.stats.device })}` : "",
    }));
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    stopMaskBusy();
    maskButton.disabled = false;
  }
}

async function loadImage(url) {
  state.image = await imageFromUrl(`${url}?t=${Date.now()}`);
  canvas.width = state.image.naturalWidth;
  canvas.height = state.image.naturalHeight;
  emptyState.style.display = "none";
  draw();
}

function draw() {
  if (!state.image) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImageForView(), 0, 0, canvas.width, canvas.height);

  if (state.points.length > 0) {
    ctx.save();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 480));
    const radius = Math.max(6, Math.round(canvas.width / 120));
    state.points.forEach((point, index) => {
      ctx.fillStyle = point.label === 1 ? "#1f84ff" : "#e5484d";
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = `${Math.max(10, radius + 3)}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(index + 1), point.x, point.y);
    });
    ctx.restore();
  }
}

function currentImageForView() {
  if (state.viewMode === "preview" && state.result) return state.result;
  if (state.viewMode === "mask" && state.overlay) return state.overlay;
  return state.image;
}

function setViewMode(mode) {
  if (mode === "mask" && !state.overlay) return;
  if (mode === "preview" && !state.result) return;
  state.viewMode = mode;
  setViewControls();
  draw();
}

function setViewControls() {
  const hasImage = Boolean(state.image);
  originalViewButton.disabled = !hasImage;
  maskViewButton.disabled = !state.overlay;
  previewViewButton.disabled = !state.result;
  downloadButton.disabled = !downloadUrlForView();

  originalViewButton.classList.toggle("active", state.viewMode === "original");
  maskViewButton.classList.toggle("active", state.viewMode === "mask");
  previewViewButton.classList.toggle("active", state.viewMode === "preview");
}

function downloadCurrentView() {
  const url = downloadUrlForView();
  if (!url || !state.imageMeta) return;

  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.imageMeta.image_id}_${state.viewMode}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function downloadUrlForView() {
  if (state.viewMode === "preview") return state.resultUrl;
  if (state.viewMode === "mask") return state.overlayUrl;
  return state.imageMeta?.url || null;
}

function setPointMode(mode) {
  state.pointMode = mode;
  includeModeButton.classList.toggle("active", mode === "include");
  excludeModeButton.classList.toggle("active", mode === "exclude");
  setStatus(mode === "include" ? t("includeMode") : t("excludeMode"));
}

function setEngine(engine) {
  state.engine = engine;
  proceduralEngineButton.classList.toggle("active", engine === "procedural");
  textureEngineButton.classList.toggle("active", engine === "texture");
  diffusionEngineButton.classList.toggle("active", engine === "diffusion");
  textureControls.hidden = engine !== "texture";
  diffusionControls.hidden = engine !== "diffusion";
  updateInpaintButtonText();
  if (engine === "diffusion") {
    setStatus(t("aiMode"));
    setPreviewStatus(t("aiPrompt"));
  } else if (engine === "texture") {
    setStatus(t("textureMode"));
    setPreviewStatus(state.materialMeta ? t("textureReady") : t("texturePrompt"));
  } else {
    setStatus(t("fastMode"));
    setPreviewStatus(t("fastReady"));
  }
}

function startPreviewBusy(engine) {
  const startedAt = Date.now();
  const baseMessage = engine === "diffusion"
    ? t("aiBusy")
    : engine === "texture"
      ? t("textureBusy")
      : t("fastBusy");

  clearPreviewTimer();
  const tick = () => {
    const seconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
    inpaintButton.textContent = engine === "diffusion" ? t("aiRunning", { seconds }) : t("generating", { seconds });
    setPreviewStatus(`${baseMessage} ${t("elapsed", { seconds })}`);
  };
  tick();
  state.previewTimer = window.setInterval(tick, 1000);
  setStatus(baseMessage);
}

function stopPreviewBusy() {
  clearPreviewTimer();
}

function startMaskBusy(startedAt) {
  clearMaskTimer();
  const tick = () => {
    const seconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
    maskButton.textContent = t("maskRunning", { seconds });
    setStatus(`${t("generatingMask")} ${t("elapsed", { seconds })}`);
  };
  tick();
  state.maskTimer = window.setInterval(tick, 1000);
}

function stopMaskBusy() {
  clearMaskTimer();
  maskButton.textContent = t("generateMask");
}

function clearMaskTimer() {
  if (state.maskTimer) {
    window.clearInterval(state.maskTimer);
    state.maskTimer = null;
  }
}

function clearPreviewTimer() {
  if (state.previewTimer) {
    window.clearInterval(state.previewTimer);
    state.previewTimer = null;
  }
}

function updateInpaintButtonText() {
  if (state.engine === "diffusion") {
    inpaintButton.textContent = t("generateAiInpaint");
  } else if (state.engine === "texture") {
    inpaintButton.textContent = t("generateTexturePreview");
  } else {
    inpaintButton.textContent = t("generateFloorPreview");
  }
}

function setPreviewStatus(message, isError = false) {
  previewStatus.textContent = message;
  previewStatus.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function labelForAlgorithm(algorithm) {
  if (algorithm === "stable_diffusion_inpainting_diffusers") return t("aiInpaintLabel");
  if (algorithm === "texture_floor_preview") return t("texturePreviewLabel");
  return t("floorPreviewLabel");
}

function pointSummary() {
  const includeCount = state.points.filter((point) => point.label === 1).length;
  const excludeCount = state.points.length - includeCount;
  return t("pointSummary", { include: includeCount, exclude: excludeCount });
}

function imageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(t("imageLoadFailed")));
    image.src = url;
  });
}

async function readJson(response) {
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(data?.detail || t("requestFailed", { status: response.status }));
  }
  return data;
}

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function applyLanguage(lang) {
  state.lang = lang;
  document.documentElement.lang = lang === "zh" ? "zh-Hant" : "en";
  for (const element of document.querySelectorAll("[data-i18n]")) {
    element.textContent = t(element.dataset.i18n);
  }
  for (const element of document.querySelectorAll("[data-i18n-placeholder]")) {
    element.placeholder = t(element.dataset.i18nPlaceholder);
  }
  zhButton.classList.toggle("active", lang === "zh");
  enButton.classList.toggle("active", lang === "en");
  if (!state.maskTimer) {
    maskButton.textContent = t("generateMask");
  }
  updateInpaintButtonText();
}

function t(key, values = {}) {
  const template = translations[state.lang]?.[key] ?? translations.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name) => values[name] ?? "");
}
