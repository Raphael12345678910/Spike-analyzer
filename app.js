import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.getElementById("video");
const videoInput = document.getElementById("videoInput");
const overlayCanvas = document.getElementById("overlayCanvas");
const ctx = overlayCanvas.getContext("2d");

const analyzeBtn = document.getElementById("analyzeBtn");
const aiBtn = document.getElementById("aiBtn");
const systemOutput = document.getElementById("systemOutput");

const angleValue = document.getElementById("angleValue");
const extensionValue = document.getElementById("extensionValue");
const reachEfficiencyValue = document.getElementById("reachEfficiencyValue");

const hittingHandInput = document.getElementById("hittingHand");

let poseLandmarker;
let drawingUtils;

let isAnalyzing = false;
let frames = [];
let lastAnalyzed = null;

// -------------------- MATH --------------------

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function round1(n) {
  return Math.round(n * 10) / 10;
}

function angle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag1 = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const mag2 = Math.sqrt(cb.x * cb.x + cb.y * cb.y);

  if (!mag1 || !mag2) return null;

  let cos = dot / (mag1 * mag2);
  cos = clamp(cos, -1, 1);

  return Math.acos(cos) * 180 / Math.PI;
}

function extensionScore(a) {
  if (!a) return null;
  return clamp(Math.round((a - 120) / 6), 1, 10);
}

function armHeight(s, w) {
  return s.y - w.y;
}

// -------------------- DRAW --------------------

function drawFrame(landmarks) {
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const cw = overlayCanvas.width;
  const ch = overlayCanvas.height;

  const va = vw / vh;
  const ca = cw / ch;

  let w, h, ox, oy;

  if (va > ca) {
    w = cw;
    h = cw / va;
    ox = 0;
    oy = (ch - h) / 2;
  } else {
    h = ch;
    w = ch * va;
    oy = 0;
    ox = (cw - w) / 2;
  }

  ctx.drawImage(video, ox, oy, w, h);

  const pts = landmarks.map(l => ({
    x: ox + l.x * w,
    y: oy + l.y * h,
    z: l.z
  }));

  drawingUtils.drawConnectors(pts, PoseLandmarker.POSE_CONNECTIONS);
  drawingUtils.drawLandmarks(pts);
}

// -------------------- LIVE RENDER --------------------

function renderLoop() {
  if (!isAnalyzing) return;

  const result = poseLandmarker.detectForVideo(video, performance.now());

  if (result.landmarks && result.landmarks.length) {
    drawFrame(result.landmarks[0]);
  }

  requestAnimationFrame(renderLoop);
}

// -------------------- FRAME CAPTURE --------------------

function captureFrames() {
  if (!isAnalyzing) return;

  if (video.ended) {
    finishAnalysis();
    return;
  }

  const result = poseLandmarker.detectForVideo(video, performance.now());

  if (result.landmarks && result.landmarks.length) {
    const lm = result.landmarks[0];

    const hand = hittingHandInput.value === "left";
    const s = lm[hand ? 11 : 12];
    const e = lm[hand ? 13 : 14];
    const w = lm[hand ? 15 : 16];

    if (s && e && w) {
      frames.push({
        t: video.currentTime,
        s,
        e,
        w,
        ang: angle(s, e, w),
        h: armHeight(s, w),
        lm
      });
    }
  }

  requestAnimationFrame(captureFrames);
}

// -------------------- ANALYSIS --------------------

function finishAnalysis() {
  isAnalyzing = false;
  video.pause();

  if (frames.length < 10) {
    systemOutput.textContent = "Not enough frames.";
    return;
  }

  const peak = [...frames].sort((a,b)=>b.h-a.h)[0];

  const contact = frames
    .filter(f => f.h > peak.h * 0.9)
    .sort((a,b)=> (b.h + b.ang/10) - (a.h + a.ang/10))[0];

  const ext = extensionScore(contact.ang);

  let eff = null;
  if (peak.h > 0) {
    eff = clamp(Math.round((contact.h / peak.h) * 10),1,10);
  }

  angleValue.textContent = round1(contact.ang) + "°";
  extensionValue.textContent = ext + "/10";
  reachEfficiencyValue.textContent = eff + "/10";

  systemOutput.textContent =
    "Analysis complete\n" +
    "Contact: " + round1(contact.t) + "s\n" +
    "Peak: " + round1(peak.t) + "s";

  video.currentTime = contact.t;
  video.onseeked = () => drawFrame(contact.lm);

  lastAnalyzed = { ext, eff };
  aiBtn.disabled = false;
}

// -------------------- UI EVENTS --------------------

analyzeBtn.onclick = () => {
  if (!video.src) return;

  frames = [];
  isAnalyzing = true;

  video.currentTime = 0;
  video.play();

  systemOutput.textContent = "Analyzing... (watch skeleton)";

  renderLoop();
  captureFrames();
};

videoInput.onchange = e => {
  const file = e.target.files[0];
  if (!file) return;

  video.src = URL.createObjectURL(file);
  video.load();
};

// -------------------- INIT --------------------

async function init() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });

  drawingUtils = new DrawingUtils(ctx);

  systemOutput.textContent = "Ready.";
}

init();