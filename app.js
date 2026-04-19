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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function round1(num) {
  return Math.round(num * 10) / 10;
}

function inchesToFeetString(totalInches) {
  if (!isFinite(totalInches)) return "--";
  const rounded = Math.round(totalInches);
  const feet = Math.floor(rounded / 12);
  const inches = rounded % 12;
  return `${feet}'${inches}"`;
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
    let wristYSum = 0;

    for (let i = Math.max(0, index - radius); i <= Math.min(frames.length - 1, index + radius); i++) {
      elbowSum += frames[i].elbowAngle;
      armRaisedSum += frames[i].armRaisedScore;
      wristYSum += frames[i].wrist.y;
      count++;
    }

    return {
      ...frame,
      elbowAngle: elbowSum / count,
      armRaisedScore: armRaisedSum / count,
      wrist: {
        ...frame.wrist,
        y: wristYSum / count
      }
    };
  });
}

function pickPeakReachFrame(frames) {
  if (!frames.length) return null;
  return [...frames].sort((a, b) => b.armRaisedScore - a.armRaisedScore)[0];
}

function pickLikelyContactFrame(frames) {
  if (!frames.length) return null;

  const sorted = [...frames].sort((a, b) => b.armRaisedScore - a.armRaisedScore);
  const top = sorted.slice(0, Math.min(5, sorted.length));

  top.sort((a, b) => {
    const aScore = a.armRaisedScore + (a.elbowAngle / 500);
    const bScore = b.armRaisedScore + (b.elbowAngle / 500);
    return bScore - aScore;
  });

  return top[0];
}

function estimateContactReach(standingReach, reachEfficiencyScore, extensionScore, repType) {
  if (!standingReach) return null;

  let maxExtraGain;
  if (repType === "standing") {
    maxExtraGain = 8;
  } else if (repType === "controlled") {
    maxExtraGain = 18;
  } else {
    maxExtraGain = 28;
  }

  const combinedQuality = ((reachEfficiencyScore || 0) * 0.65) + ((extensionScore || 0) * 0.35);
  const gain = (combinedQuality / 10) * maxExtraGain;

  return {
    gainAboveStandingReach: round1(gain),
    estimatedContactReach: round1(standingReach + gain)
  };
}

function buildSystemNotes({
  cameraAngle,
  repType,
  standingReach,
  netHeight,
  framesCount,
  extensionScore,
  reachEfficiencyScore
}) {
  const lines = [];

  lines.push(`Frames analyzed: ${framesCount}`);
  lines.push(`Clip type: ${repType}`);
  lines.push(`Camera angle: ${cameraAngle}`);

  if (cameraAngle !== "side") {
    lines.push("Warning: non-side view lowers measurement trust.");
  }

  if (!standingReach) {
    lines.push("Standing reach not provided: reach estimates are unavailable.");
  }

  if (!netHeight) {
    lines.push("Net height not provided: margin above net is unavailable.");
  }

  if (extensionScore !== null && extensionScore <= 4) {
    lines.push("Main mechanical flag: arm stays noticeably bent near likely contact.");
  }

  if (reachEfficiencyScore !== null && reachEfficiencyScore <= 5) {
    lines.push("Main reach flag: the likely contact frame is well below this clip’s peak reachable frame.");
  }

  lines.push("No ball tracking yet. Likely contact frame is estimated, not confirmed.");

  return lines.join("\n");
}

function buildAiPayload() {
  if (!lastAnalyzed) return null;

  const warnings = [];

  if (cameraAngleInput.value !== "side") {
    warnings.push("Non-side view reduces reliability");
  }

  if (!standingReachInput.value) {
    warnings.push("No standing reach provided");
  }

  if (!netHeightInput.value) {
    warnings.push("No net height provided");
  }

  warnings.push("No ball tracking");
  warnings.push("Likely contact frame is estimated");

  return {
    hittingHand: hittingHandInput.value,
    cameraAngle: cameraAngleInput.value,
    repType: repTypeInput.value,
    userNotes: userNotesInput.value || "",
    elbowAngle: lastAnalyzed.elbowAngle !== null ? `${round1(lastAnalyzed.elbowAngle)}°` : "unknown",
    extensionScore: lastAnalyzed.extensionScore ?? "unknown",
    reachEfficiencyScore: lastAnalyzed.reachEfficiencyScore ?? "unknown",
    estimatedContactReach: lastAnalyzed.estimatedContactReach !== null
      ? inchesToFeetString(lastAnalyzed.estimatedContactReach)
      : "not available",
    gainAboveStandingReach: lastAnalyzed.gainAboveStandingReach !== null
      ? `${round1(lastAnalyzed.gainAboveStandingReach)} in`
      : "not available",
    marginAboveNet: lastAnalyzed.marginAboveNet !== null
      ? `${round1(lastAnalyzed.marginAboveNet)} in`
      : "not available",
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

function resetUI() {
  angleValue.textContent = "--";
  extensionValue.textContent = "--";
  reachEfficiencyValue.textContent = "--";
  contactReachValue.textContent = "--";
  gainValue.textContent = "--";
  netMarginValue.textContent = "--";
  systemOutput.textContent = "No analysis yet.";
  aiCoach.textContent = "Run the analysis first, then click “Get AI Coaching.”";
  aiBtn.disabled = true;
  lastAnalyzed = null;

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
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
  if (!video.src) {
    systemOutput.textContent = "Upload a video first.";
    return;
  }

  if (!poseLandmarker) {
    systemOutput.textContent = "Pose model is still loading.";
    return;
  }

  video.pause();

  await new Promise((resolve) => {
    if (video.readyState >= 2) {
      resolve();
    } else {
      video.onloadeddata = () => resolve();
    }
  });

  overlayCanvas.width = video.videoWidth;
  overlayCanvas.height = video.videoHeight;

  const frames = [];
  const hand = hittingHandInput.value;
  const shoulderIndex = hand === "left" ? 11 : 12;
  const elbowIndex = hand === "left" ? 13 : 14;
  const wristIndex = hand === "left" ? 15 : 16;

  const duration = video.duration;
  const sampleRate = 12;

  async function seekTo(time) {
    return new Promise((resolve) => {
      const target = Math.max(0.01, Math.min(time, duration - 0.001));

      if (Math.abs(video.currentTime - target) < 0.01) {
        resolve();
        return;
      }

      const handler = () => {
        video.removeEventListener("seeked", handler);
        resolve();
      };

      video.addEventListener("seeked", handler, { once: true });
      video.currentTime = target;
    });
  }

  try {
    systemOutput.textContent = "Analyzing clip...";

    await seekTo(0.01);

    for (let t = 0.01; t < duration; t += 1 / sampleRate) {
      await seekTo(t);

      const result = poseLandmarker.detectForVideo(video, performance.now());

      if (!result.landmarks || !result.landmarks.length) continue;

      const landmarks = result.landmarks[0];
      const shoulder = landmarks[shoulderIndex];
      const elbow = landmarks[elbowIndex];
      const wrist = landmarks[wristIndex];

      if (!shoulder || !elbow || !wrist) continue;

      const elbowAngle = angleBetweenThreePoints(shoulder, elbow, wrist);
      const armRaisedScore = getArmRaisedScore(shoulder, wrist);

      frames.push({
        time: t,
        shoulder,
        elbow,
        wrist,
        elbowAngle,
        armRaisedScore,
        landmarks
      });
    }

    if (frames.length < 5) {
      systemOutput.textContent = "Could not detect enough body frames. Try a cleaner side-view clip.";
      return;
    }

    const smoothed = smoothFrames(frames, 2);
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

    const standingReach = Number(standingReachInput.value) || null;
    const netHeight = Number(netHeightInput.value) || null;
    const reachEstimate = estimateContactReach(
      standingReach,
      reachEfficiencyScore,
      extensionScore,
      repTypeInput.value
    );

    const estimatedContactReach = reachEstimate ? reachEstimate.estimatedContactReach : null;
    const gainAboveStandingReach = reachEstimate ? reachEstimate.gainAboveStandingReach : null;
    const marginAboveNet =
      reachEstimate && netHeight ? round1(reachEstimate.estimatedContactReach - netHeight) : null;

    lastAnalyzed = {
      elbowAngle: contactFrame.elbowAngle,
      extensionScore,
      reachEfficiencyScore,
      estimatedContactReach,
      gainAboveStandingReach,
      marginAboveNet,
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

    contactReachValue.textContent = estimatedContactReach !== null
      ? inchesToFeetString(estimatedContactReach)
      : "Need standing reach";

    gainValue.textContent = gainAboveStandingReach !== null
      ? `+${round1(gainAboveStandingReach)} in`
      : "Need standing reach";

    netMarginValue.textContent = marginAboveNet !== null
      ? `${marginAboveNet >= 0 ? "+" : ""}${round1(marginAboveNet)} in`
      : "Need net height";

    systemOutput.textContent = buildSystemNotes({
      cameraAngle: cameraAngleInput.value,
      repType: repTypeInput.value,
      standingReach,
      netHeight,
      framesCount: smoothed.length,
      extensionScore,
      reachEfficiencyScore
    });

    aiBtn.disabled = false;

    await seekTo(contactFrame.time);
    drawFrame(contactFrame.landmarks);

  } catch (error) {
    console.error(error);
    systemOutput.textContent = "Analysis failed. Open the browser console and send me the error.";
  }
}

function drawFrame(landmarks) {
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  ctx.drawImage(video, 0, 0, overlayCanvas.width, overlayCanvas.height);

  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    lineWidth: 3
  });

  drawingUtils.drawLandmarks(landmarks, {
    radius: 4
  });
}

await createPoseModel();
systemOutput.textContent = "Model loaded. Upload a clip, then click “Analyze Video.”";