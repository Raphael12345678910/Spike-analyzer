import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const videoInput = document.getElementById("videoInput");
const video = document.getElementById("video");
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
const standingReachInput = document.getElementById("standingReach");
const netHeightInput = document.getElementById("netHeight");
const userNotesInput = document.getElementById("userNotes");

let poseLandmarker;
let drawingUtils;
let lastAnalyzed = null;
let currentVideoUrl = null;
let isAnalyzing = false;
let collectedFrames = [];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function round1(num) {
  return Math.round(num * 10) / 10;
}

function angleBetweenThreePoints(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const magCB = Math.sqrt(cb.x * cb.x + cb.y * cb.y);

  if (magAB === 0 || magCB === 0) return null;

  let cosine = dot / (magAB * magCB);
  cosine = clamp(cosine, -1, 1);

  return Math.acos(cosine) * (180 / Math.PI);
}

function getExtensionScore(elbowAngle) {
  if (!isFinite(elbowAngle)) return null;
  const normalized = (elbowAngle - 120) / 60;
  return clamp(Math.round(normalized * 10), 1, 10);
}

function getArmRaisedScore(shoulder, wrist) {
  return shoulder.y - wrist.y;
}

function smoothFrames(frames, radius = 2) {
  return frames.map((frame, index) => {
    let count = 0;
    let elbowSum = 0;
    let armRaisedSum = 0;

    for (let i = Math.max(0, index - radius); i <= Math.min(frames.length - 1, index + radius); i++) {
      elbowSum += frames[i].elbowAngle;
      armRaisedSum += frames[i].armRaisedScore;
      count++;
    }

    return {
      ...frame,
      elbowAngle: elbowSum / count,
      armRaisedScore: armRaisedSum / count
    };
  });
}

function pickPeakReachFrame(frames) {
  if (!frames.length) return null;
  return [...frames].sort((a, b) => b.armRaisedScore - a.armRaisedScore)[0];
}

function pickLikelyContactFrame(frames) {
  if (!frames.length) return null;

  const peak = pickPeakReachFrame(frames);
  if (!peak) return null;

  const candidates = frames.filter(
    (f) => f.armRaisedScore >= peak.armRaisedScore * 0.9
  );

  if (!candidates.length) return peak;

  candidates.sort((a, b) => {
    const aScore =
      (a.armRaisedScore * 0.6) +
      ((a.elbowAngle || 0) * 0.4);

    const bScore =
      (b.armRaisedScore * 0.6) +
      ((b.elbowAngle || 0) * 0.4);

    return bScore - aScore;
  });

  return candidates[0];
}

function drawFrame(landmarks) {
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const canvasW = overlayCanvas.width;
  const canvasH = overlayCanvas.height;
  const videoW = video.videoWidth;
  const videoH = video.videoHeight;

  const videoAspect = videoW / videoH;
  const canvasAspect = canvasW / canvasH;

  let drawW, drawH, offsetX, offsetY;

  if (videoAspect > canvasAspect) {
    drawW = canvasW;
    drawH = canvasW / videoAspect;
    offsetX = 0;
    offsetY = (canvasH - drawH) / 2;
  } else {
    drawH = canvasH;
    drawW = canvasH * videoAspect;
    offsetY = 0;
    offsetX = (canvasW - drawW) / 2;
  }

  ctx.drawImage(video, offsetX, offsetY, drawW, drawH);

  const transformedLandmarks = landmarks.map((lm) => ({
    x: offsetX + lm.x * drawW,
    y: offsetY + lm.y * drawH,
    z: lm.z,
    visibility: lm.visibility
  }));

  drawingUtils.drawConnectors(
    transformedLandmarks,
    PoseLandmarker.POSE_CONNECTIONS,
    { lineWidth: 3 }
  );

  drawingUtils.drawLandmarks(transformedLandmarks, {
    radius: 4
  });
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
  lastAnalyzed = null;
  collectedFrames = [];
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function buildSystemNotes({
  cameraAngle,
  repType,
  framesCount,
  extensionScore,
  reachEfficiencyScore,
  contactTime,
  peakTime
}) {
  const lines = [];

  lines.push(`Frames analyzed: ${framesCount}`);
  lines.push(`Clip type: ${repType}`);
  lines.push(`Camera angle: ${cameraAngle}`);

  if (cameraAngle !== "side") {
    lines.push("Warning: non-side view lowers trust.");
  }

  if (extensionScore !== null && extensionScore <= 4) {
    lines.push("Main issue: arm stays noticeably bent near likely contact.");
  }

  if (reachEfficiencyScore !== null && reachEfficiencyScore <= 5) {
    lines.push("Main issue: likely contact is well below this clip’s own peak reach frame.");
  }

  lines.push("No ball tracking yet. Likely contact frame is estimated.");
  lines.push("Reach / vertical estimates are temporarily disabled because they were too unreliable.");

  if (isFinite(contactTime)) {
    lines.push(`Likely contact frame time: ${round1(contactTime)}s`);
  }

  if (isFinite(peakTime)) {
    lines.push(`Peak reach frame time: ${round1(peakTime)}s`);
  }

  return lines.join("\n");
}

function buildAiPayload() {
  if (!lastAnalyzed) return null;

  const warnings = [];

  if (cameraAngleInput.value !== "side") {
    warnings.push("Non-side view reduces reliability");
  }

  warnings.push("No ball tracking");
  warnings.push("Likely contact frame is estimated");
  warnings.push("Reach estimates temporarily disabled because measurement quality was poor");

  return {
    hittingHand: hittingHandInput.value,
    cameraAngle: cameraAngleInput.value,
    repType: repTypeInput.value,
    userNotes: userNotesInput.value || "",
    elbowAngle: lastAnalyzed.elbowAngle !== null ? `${round1(lastAnalyzed.elbowAngle)}°` : "unknown",
    extensionScore: lastAnalyzed.extensionScore ?? "unknown",
    reachEfficiencyScore: lastAnalyzed.reachEfficiencyScore ?? "unknown",
    estimatedContactReach: "not available",
    gainAboveStandingReach: "not available",
    marginAboveNet: "not available",
    warnings
  };
}

async function createPoseModel() {
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
}

videoInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
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

  if (isAnalyzing) return;

  aiBtn.disabled = true;
  aiCoach.textContent = "Run the analysis first, then click “Get AI Coaching.”";
  systemOutput.textContent = "Analyzing clip...";
  await analyzeVideo();
});

aiBtn.addEventListener("click", async () => {
  const payload = buildAiPayload();

  if (!payload) {
    aiCoach.textContent = "Run the analysis first.";
    return;
  }

  aiCoach.textContent = "Getting AI coaching...";

  try {
    const response = await fetch("/api/coach", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error("Backend request failed");
    }

    const result = await response.json();
    aiCoach.textContent = result.feedback || "No AI coaching returned.";
  } catch (error) {
    console.error(error);
    aiCoach.textContent = "AI coaching failed to load.";
  }
});

async function analyzeVideo() {
  video.pause();
  video.currentTime = 0;

  await new Promise((resolve) => {
    if (video.readyState >= 2) {
      resolve();
    } else {
      video.onloadeddata = () => resolve();
    }
  });

  overlayCanvas.width = video.videoWidth;
  overlayCanvas.height = video.videoHeight;

  collectedFrames = [];
  isAnalyzing = true;

  const hand = hittingHandInput.value;
  const shoulderIndex = hand === "left" ? 11 : 12;
  const elbowIndex = hand === "left" ? 13 : 14;
  const wristIndex = hand === "left" ? 15 : 16;

  return new Promise((resolve) => {
    function processPlaybackFrame() {
      if (!isAnalyzing) {
        resolve();
        return;
      }

      if (video.paused || video.ended) {
        finishAnalysis();
        resolve();
        return;
      }

      const result = poseLandmarker.detectForVideo(video, performance.now());

      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      if (result.landmarks && result.landmarks.length > 0) {
        const landmarks = result.landmarks[0];
        const shoulder = landmarks[shoulderIndex];
        const elbow = landmarks[elbowIndex];
        const wrist = landmarks[wristIndex];

        if (shoulder && elbow && wrist) {
          const elbowAngle = angleBetweenThreePoints(shoulder, elbow, wrist);
          const armRaisedScore = getArmRaisedScore(shoulder, wrist);

          collectedFrames.push({
            time: video.currentTime,
            shoulder,
            elbow,
            wrist,
            elbowAngle,
            armRaisedScore,
            landmarks
          });
        }

        drawFrame(landmarks);
      }

      requestAnimationFrame(processPlaybackFrame);
    }

    function finishAnalysis() {
      isAnalyzing = false;
      video.pause();

      if (collectedFrames.length < 5) {
        systemOutput.textContent = "Could not detect enough body frames. Try a cleaner side-view clip.";
        return;
      }

      const smoothed = smoothFrames(collectedFrames, 2);
      const peakFrame = pickPeakReachFrame(smoothed);
      const contactFrame = pickLikelyContactFrame(smoothed);

      if (!peakFrame || !contactFrame) {
        systemOutput.textContent = "Could not identify a usable reach/contact frame.";
        return;
      }

      const extensionScore = getExtensionScore(contactFrame.elbowAngle);

      let reachEfficiencyScore = null;
      if (peakFrame.armRaisedScore > 0) {
        reachEfficiencyScore = clamp(
          Math.round((contactFrame.armRaisedScore / peakFrame.armRaisedScore) * 10),
          1,
          10
        );
      }

      lastAnalyzed = {
        elbowAngle: contactFrame.elbowAngle,
        extensionScore,
        reachEfficiencyScore,
        estimatedContactReach: null,
        gainAboveStandingReach: null,
        marginAboveNet: null,
        contactTime: contactFrame.time,
        peakTime: peakFrame.time
      };

      angleValue.textContent = lastAnalyzed.elbowAngle !== null
        ? `${round1(lastAnalyzed.elbowAngle)}°`
        : "--";

      extensionValue.textContent = extensionScore !== null
        ? `${extensionScore}/10`
        : "--";

      reachEfficiencyValue.textContent = reachEfficiencyScore !== null
        ? `${reachEfficiencyScore}/10`
        : "--";

      contactReachValue.textContent = "Disabled for now";
      gainValue.textContent = "Disabled for now";
      netMarginValue.textContent = "Disabled for now";

      systemOutput.textContent = buildSystemNotes({
        cameraAngle: cameraAngleInput.value,
        repType: repTypeInput.value,
        framesCount: smoothed.length,
        extensionScore,
        reachEfficiencyScore,
        contactTime: lastAnalyzed.contactTime,
        peakTime: lastAnalyzed.peakTime
      });

      aiBtn.disabled = false;

      video.currentTime = contactFrame.time;
      video.onseeked = () => {
        drawFrame(contactFrame.landmarks);
        video.onseeked = null;
      };
    }

    video.play();
    requestAnimationFrame(processPlaybackFrame);
  });
}

await createPoseModel();
systemOutput.textContent = "Model loaded. Upload a clip, then click “Analyze Video.”";