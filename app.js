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
const resetBtn = document.getElementById("resetBtn");
const aiBtn = document.getElementById("aiBtn");
const aiCoach = document.getElementById("aiCoach");

const angleValue = document.getElementById("angleValue");
const extensionValue = document.getElementById("extensionValue");
const reachEfficiencyValue = document.getElementById("reachEfficiencyValue");
const contactReachValue = document.getElementById("contactReachValue");
const gainValue = document.getElementById("gainValue");
const netMarginValue = document.getElementById("netMarginValue");
const systemOutput = document.getElementById("systemOutput");

const hittingHandInput = document.getElementById("hittingHand");
const cameraAngleInput = document.getElementById("cameraAngle");
const repTypeInput = document.getElementById("repType");
const userNotesInput = document.getElementById("userNotes");

let poseLandmarker;
let drawingUtils;

let isAnalyzing = false;
let frames = [];
let lastAnalyzed = null;
let currentVideoUrl = null;
let latestLandmarks = null;

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

function drawFrame(landmarks) {
  if (!landmarks) return;

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const cw = overlayCanvas.width;
  const ch = overlayCanvas.height;

  if (!vw || !vh || !cw || !ch) return;

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

  const pts = landmarks.map((l) => ({
    x: ox + l.x * w,
    y: oy + l.y * h,
    z: l.z,
    visibility: l.visibility
  }));

  drawingUtils.drawConnectors(pts, PoseLandmarker.POSE_CONNECTIONS, {
    lineWidth: 3
  });

  drawingUtils.drawLandmarks(pts, {
    radius: 4
  });
}

function renderLoop() {
  if (!isAnalyzing) return;

  const result = poseLandmarker.detectForVideo(video, performance.now());

  if (result.landmarks && result.landmarks.length > 0) {
    latestLandmarks = result.landmarks[0];
    drawFrame(latestLandmarks);
  }

  requestAnimationFrame(renderLoop);
}

function captureFrames() {
  if (!isAnalyzing) return;

  if (video.ended) {
    finishAnalysis();
    return;
  }

  const result = poseLandmarker.detectForVideo(video, performance.now());

  if (result.landmarks && result.landmarks.length > 0) {
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

function finishAnalysis() {
  isAnalyzing = false;
  video.pause();

  if (frames.length < 10) {
    systemOutput.textContent = "Not enough frames. Try a cleaner side-view clip.";
    return;
  }

  const peak = [...frames].sort((a, b) => b.h - a.h)[0];

  const contactCandidates = frames.filter((f) => f.h > peak.h * 0.9);

  const contact = (contactCandidates.length ? contactCandidates : frames)
    .sort((a, b) => (b.h + b.ang / 10) - (a.h + a.ang / 10))[0];

  const ext = extensionScore(contact.ang);

  let eff = null;
  if (peak.h > 0) {
    eff = clamp(Math.round((contact.h / peak.h) * 10), 1, 10);
  }

  angleValue.textContent = round1(contact.ang) + "°";
  extensionValue.textContent = ext + "/10";
  reachEfficiencyValue.textContent = eff + "/10";

  contactReachValue.textContent = "Disabled for now";
  gainValue.textContent = "Disabled for now";
  netMarginValue.textContent = "Disabled for now";

  systemOutput.textContent =
    "Analysis complete\n" +
    "Likely contact frame: " + round1(contact.t) + "s\n" +
    "Peak reach frame: " + round1(peak.t) + "s\n" +
    "No ball tracking yet. Reach / net estimates are disabled for now because they were not trustworthy.";

  lastAnalyzed = {
    elbowAngle: contact.ang,
    ext,
    eff,
    time: contact.t
  };

  aiBtn.disabled = false;

  video.currentTime = contact.t;

  const handleSeek = () => {
    drawFrame(contact.lm);
    video.removeEventListener("seeked", handleSeek);
  };

  video.addEventListener("seeked", handleSeek);
}

function resetUI() {
  angleValue.textContent = "--";
  extensionValue.textContent = "--";
  reachEfficiencyValue.textContent = "--";
  contactReachValue.textContent = "Disabled for now";
  gainValue.textContent = "Disabled for now";
  netMarginValue.textContent = "Disabled for now";
  systemOutput.textContent = "No analysis yet.";
  aiCoach.textContent = "Run the analysis first, then click “Get AI Coaching.”";
  aiBtn.disabled = true;

  isAnalyzing = false;
  frames = [];
  lastAnalyzed = null;
  latestLandmarks = null;

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function resizeCanvasToVideoBox() {
  const rect = video.getBoundingClientRect();
  overlayCanvas.width = Math.max(1, Math.floor(rect.width));
  overlayCanvas.height = Math.max(1, Math.floor(rect.height));
}

videoInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (currentVideoUrl) {
    URL.revokeObjectURL(currentVideoUrl);
  }

  currentVideoUrl = URL.createObjectURL(file);
  video.src = currentVideoUrl;
  video.load();

  resetUI();
  systemOutput.textContent = "Video loaded. Click “Analyze Video” when ready.";
});

resetBtn.addEventListener("click", () => {
  if (currentVideoUrl) {
    URL.revokeObjectURL(currentVideoUrl);
    currentVideoUrl = null;
  }

  video.removeAttribute("src");
  video.load();
  videoInput.value = "";
  resetUI();
});

analyzeBtn.addEventListener("click", async () => {
  if (!video.src) {
    systemOutput.textContent = "Upload a video first.";
    return;
  }

  if (!poseLandmarker) {
    systemOutput.textContent = "Pose model is still loading.";
    return;
  }

  frames = [];
  latestLandmarks = null;
  lastAnalyzed = null;
  aiBtn.disabled = true;
  aiCoach.textContent = "Run the analysis first, then click “Get AI Coaching.”";

  resizeCanvasToVideoBox();

  isAnalyzing = true;
  video.currentTime = 0;
  video.play();

  systemOutput.textContent = "Analyzing... watch the skeleton overlay.";

  renderLoop();
  captureFrames();
});

aiBtn.addEventListener("click", async () => {
  if (!lastAnalyzed) {
    aiCoach.textContent = "Run analysis first.";
    return;
  }

  aiCoach.textContent = "Loading...";

  try {
    const res = await fetch("/api/coach", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        hittingHand: hittingHandInput.value,
        cameraAngle: cameraAngleInput.value,
        repType: repTypeInput.value,
        userNotes: userNotesInput.value || "",
        elbowAngle: lastAnalyzed.elbowAngle,
        extensionScore: lastAnalyzed.ext,
        reachEfficiencyScore: lastAnalyzed.eff,
        estimatedContactReach: "not available",
        gainAboveStandingReach: "not available",
        marginAboveNet: "not available",
        warnings: ["no ball tracking", "likely contact frame is estimated"]
      })
    });

    const data = await res.json();
    aiCoach.textContent = data.feedback || "No response.";
  } catch (err) {
    console.error(err);
    aiCoach.textContent = "AI failed.";
  }
});

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
  systemOutput.textContent = "Ready. Upload a clip, then click “Analyze Video.”";
}

window.addEventListener("resize", () => {
  if (video.src) {
    resizeCanvasToVideoBox();
    if (latestLandmarks) {
      drawFrame(latestLandmarks);
    }
  }
});

init();