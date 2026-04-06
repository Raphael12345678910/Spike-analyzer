import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const videoInput = document.getElementById("videoInput");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const output = document.getElementById("output");

const angleValue = document.getElementById("angleValue");
const extensionValue = document.getElementById("extensionValue");
const contactValue = document.getElementById("contactValue");
const reachValue = document.getElementById("reachValue");
const gainValue = document.getElementById("gainValue");
const netMarginValue = document.getElementById("netMarginValue");

const hittingHandInput = document.getElementById("hittingHand");
const heightFeetInput = document.getElementById("heightFeet");
const heightInchesInput = document.getElementById("heightInches");
const reachFeetInput = document.getElementById("reachFeet");
const reachInchesInput = document.getElementById("reachInches");
const netFeetInput = document.getElementById("netFeet");
const netInchesInput = document.getElementById("netInches");
const cameraAngleInput = document.getElementById("cameraAngle");
const repTypeInput = document.getElementById("repType");
const userNotesInput = document.getElementById("userNotes");

const startCalibrationBtn = document.getElementById("startCalibrationBtn");
const resetCalibrationBtn = document.getElementById("resetCalibrationBtn");
const calibrationStatus = document.getElementById("calibrationStatus");

let poseLandmarker;
let drawingUtils;
let bestResult = null;

let smoothedPoints = {
  shoulder: null,
  elbow: null,
  wrist: null,
  nose: null
};

let calibrationMode = false;
let calibrationStep = 0;
let netCalibration = {
  netTop: null,
  floorPoint: null,
  pixelsPerInch: null
};

function feetInchesToInches(feet, inches) {
  const f = Number(feet) || 0;
  const i = Number(inches) || 0;
  return (f * 12) + i;
}

function inchesToFeetString(totalInches) {
  if (!isFinite(totalInches) || totalInches <= 0) return "--";
  const rounded = Math.round(totalInches);
  const feet = Math.floor(rounded / 12);
  const inches = rounded % 12;
  return `${feet}'${inches}"`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function smoothPoint(oldPoint, newPoint, alpha = 0.7) {
  if (!oldPoint) return { ...newPoint };

  return {
    x: alpha * oldPoint.x + (1 - alpha) * newPoint.x,
    y: alpha * oldPoint.y + (1 - alpha) * newPoint.y,
    z: alpha * (oldPoint.z || 0) + (1 - alpha) * (newPoint.z || 0)
  };
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
  if (elbowAngle === null || !isFinite(elbowAngle)) return null;
  const normalized = (elbowAngle - 120) / 60;
  return clamp(Math.round(normalized * 10), 1, 10);
}

function getContactHeightScore(wristY, noseY, shoulderY) {
  const headDiff = noseY - wristY;
  const shoulderDiff = shoulderY - wristY;
  const weighted = headDiff * 0.7 + shoulderDiff * 0.3;
  const normalized = weighted / 0.18;
  return clamp(Math.round(normalized * 10), 1, 10);
}

function getContactHeightLabel(score) {
  if (score <= 3) return "Low for attacking reach";
  if (score <= 5) return "Usable but low";
  if (score <= 7) return "Decent, but not max reach";
  if (score <= 9) return "Strong attacking height";
  return "Near max reach zone";
}

function resetCalibration() {
  calibrationMode = false;
  calibrationStep = 0;
  netCalibration = {
    netTop: null,
    floorPoint: null,
    pixelsPerInch: null
  };
  calibrationStatus.textContent = "Calibration not set.";
}

function drawCalibrationMarkers() {
  if (netCalibration.netTop) {
    ctx.beginPath();
    ctx.arc(netCalibration.netTop.x, netCalibration.netTop.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#22c55e";
    ctx.fill();
  }

  if (netCalibration.floorPoint) {
    ctx.beginPath();
    ctx.arc(netCalibration.floorPoint.x, netCalibration.floorPoint.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#f97316";
    ctx.fill();
  }

  if (netCalibration.netTop && netCalibration.floorPoint) {
    ctx.beginPath();
    ctx.moveTo(netCalibration.netTop.x, netCalibration.netTop.y);
    ctx.lineTo(netCalibration.floorPoint.x, netCalibration.floorPoint.y);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function updateCalibrationScale() {
  const netHeightInches = feetInchesToInches(netFeetInput.value, netInchesInput.value);

  if (!netHeightInches || !netCalibration.netTop || !netCalibration.floorPoint) {
    netCalibration.pixelsPerInch = null;
    return;
  }

  const pixelHeight = Math.abs(netCalibration.floorPoint.y - netCalibration.netTop.y);
  if (pixelHeight <= 0) {
    netCalibration.pixelsPerInch = null;
    return;
  }

  netCalibration.pixelsPerInch = pixelHeight / netHeightInches;
}

function estimateContactReachFromCalibration(wristPixelY, standingReachInches, netHeightInches) {
  if (
    !netCalibration.pixelsPerInch ||
    !netCalibration.floorPoint ||
    !netHeightInches
  ) {
    return null;
  }

  const contactHeightFromFloorPx = netCalibration.floorPoint.y - wristPixelY;
  const contactReachInches = contactHeightFromFloorPx / netCalibration.pixelsPerInch;

  const gainAboveStandingReach = standingReachInches
    ? contactReachInches - standingReachInches
    : null;

  const marginAboveNet = contactReachInches - netHeightInches;

  return {
    estimatedContactReach: contactReachInches,
    gainAboveStandingReach,
    marginAboveNet
  };
}

function buildFeedback({
  elbowAngle,
  extensionScore,
  contactHeightScore,
  contactHeightLabel,
  calibratedReach,
  cameraAngle,
  repType,
  userNotes,
  hasStandingReach,
  hasNetHeight,
  hasCalibration
}) {
  const parts = [];

  if (elbowAngle !== null) {
    if (extensionScore <= 4) {
      parts.push(
        `Your elbow was about ${elbowAngle.toFixed(0)}°, which is only around ${extensionScore}/10 toward full extension. You are probably losing reach and power by staying too bent near likely contact.`
      );
    } else if (extensionScore <= 7) {
      parts.push(
        `Your elbow was about ${elbowAngle.toFixed(0)}°, or roughly ${extensionScore}/10 toward full extension. That is usable, but still short of what looks like your best reach.`
      );
    } else {
      parts.push(
        `Your elbow was about ${elbowAngle.toFixed(0)}°, which is roughly ${extensionScore}/10 toward full extension. Your arm position looks much closer to a strong attacking shape.`
      );
    }
  }

  parts.push(
    `Your contact height scored about ${contactHeightScore}/10 for attacking reach, which fits the "${contactHeightLabel}" range.`
  );

  if (calibratedReach) {
    parts.push(
      `Using your optional net calibration, your estimated contact reach in this clip was about ${inchesToFeetString(calibratedReach.estimatedContactReach)}.`
    );

    if (calibratedReach.gainAboveStandingReach !== null && isFinite(calibratedReach.gainAboveStandingReach)) {
      parts.push(
        `That is about ${Math.round(calibratedReach.gainAboveStandingReach)} inches above your standing reach, which acts like a rough contact-reach gain estimate.`
      );
    }

    if (isFinite(calibratedReach.marginAboveNet)) {
      if (calibratedReach.marginAboveNet < 0) {
        parts.push(
          `Your estimated contact point appears below the net height reference, which likely means the estimate or the clip setup is weak, or that the selected frame was not truly near contact.`
        );
      } else if (calibratedReach.marginAboveNet < 8) {
        parts.push(
          `Your estimated contact point was only slightly above the net, so you may still be contacting lower than ideal for a stronger attacking angle.`
        );
      } else {
        parts.push(
          `Your estimated contact point was comfortably above the net, which is a much stronger attacking position.`
        );
      }
    }
  } else {
    if (hasStandingReach || hasNetHeight) {
      parts.push(
        `You entered optional measurement info, but the app still needs net calibration clicks to estimate contact reach and vertical-style numbers.`
      );
    } else {
      parts.push(
        `If you want rough reach and vertical-style estimates, enter standing reach and net height, then use net calibration on the video.`
      );
    }
  }

  if (cameraAngle === "front") {
    parts.push(
      `Front-view clips are much less trustworthy for angle and reach estimates. A true side view will make this app more useful.`
    );
  } else if (cameraAngle === "diagonal") {
    parts.push(
      `A slight diagonal can still work, but it introduces more measurement error than a true side view.`
    );
  }

  if (repType === "standing") {
    parts.push(
      `This was marked as a standing rep, so any reach estimate here should not be treated like full-approach max jump contact.`
    );
  }

  if (userNotes && userNotes.trim().length > 0) {
    const lower = userNotes.toLowerCase();

    if (lower.includes("low")) {
      parts.push(
        `You mentioned feeling low, and the current contact height score does support that as a likely issue.`
      );
    }

    if (lower.includes("late")) {
      parts.push(
        `You mentioned feeling late. This version still does not truly track the ball, so timing feedback is still limited.`
      );
    }

    if (lower.includes("bent") || lower.includes("arm")) {
      parts.push(
        `You mentioned your arm, and the extension score does suggest that arm shape is one of the main things worth watching here.`
      );
    }
  }

  if (!hasStandingReach) {
    parts.push(
      `You did not enter standing reach, so the app cannot estimate gain above standing reach yet.`
    );
  }

  if (!hasNetHeight) {
    parts.push(
      `You did not enter net height, so the app cannot estimate your margin above the net yet.`
    );
  }

  if ((hasStandingReach || hasNetHeight) && !hasCalibration) {
    parts.push(
      `To improve measurement accuracy, use a clip where the top of the net and the floor directly below it are visible, then set the two calibration clicks.`
    );
  }

  return parts.join("\n\n");
}

async function createPoseLandmarker() {
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

async function analyzeVideo() {
  if (!poseLandmarker) {
    output.textContent = "Pose model is still loading.";
    return;
  }

  const hittingHand = hittingHandInput.value;
  const heightInches = feetInchesToInches(heightFeetInput.value, heightInchesInput.value);
  const standingReachInches = feetInchesToInches(reachFeetInput.value, reachInchesInput.value);
  const netHeightInches = feetInchesToInches(netFeetInput.value, netInchesInput.value);
  const cameraAngle = cameraAngleInput.value;
  const repType = repTypeInput.value;
  const userNotes = userNotesInput.value || "";

  const hasStandingReach = standingReachInches > 0;
  const hasNetHeight = netHeightInches > 0;
  const hasCalibration = !!netCalibration.pixelsPerInch;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  let lastTime = -1;
  bestResult = null;

  smoothedPoints = {
    shoulder: null,
    elbow: null,
    wrist: null,
    nose: null
  };

  async function processFrame() {
    if (video.paused || video.ended) {
      if (bestResult) {
        angleValue.textContent = bestResult.elbowAngle
          ? `${bestResult.elbowAngle.toFixed(1)}°`
          : "--";

        extensionValue.textContent = bestResult.extensionScore
          ? `${bestResult.extensionScore}/10 toward full extension`
          : "--";

        contactValue.textContent = `${bestResult.contactHeightScore}/10 — ${bestResult.contactHeightLabel}`;

        if (bestResult.calibratedReach) {
          reachValue.textContent = inchesToFeetString(bestResult.calibratedReach.estimatedContactReach);

          gainValue.textContent =
            bestResult.calibratedReach.gainAboveStandingReach !== null &&
            isFinite(bestResult.calibratedReach.gainAboveStandingReach)
              ? `${Math.round(bestResult.calibratedReach.gainAboveStandingReach) >= 0 ? "+" : ""}${Math.round(bestResult.calibratedReach.gainAboveStandingReach)} in`
              : "Enter standing reach";

          netMarginValue.textContent =
            isFinite(bestResult.calibratedReach.marginAboveNet)
              ? `${Math.round(bestResult.calibratedReach.marginAboveNet) >= 0 ? "+" : ""}${Math.round(bestResult.calibratedReach.marginAboveNet)} in`
              : "Enter net height";
        } else {
          reachValue.textContent = "Optional: add standing reach + net calibration";
          gainValue.textContent = hasStandingReach ? "Add net calibration" : "Enter standing reach";
          netMarginValue.textContent = hasNetHeight ? "Add net calibration" : "Enter net height";
        }

        output.textContent = `Main feedback:\n\n${bestResult.feedback}`;
      } else {
        angleValue.textContent = "--";
        extensionValue.textContent = "--";
        contactValue.textContent = "--";
        reachValue.textContent = "--";
        gainValue.textContent = "--";
        netMarginValue.textContent = "--";
        output.textContent = "No pose detected clearly enough.";
      }
      return;
    }

    if (video.currentTime !== lastTime) {
      lastTime = video.currentTime;

      const result = poseLandmarker.detectForVideo(video, performance.now());

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (result.landmarks && result.landmarks.length > 0) {
        const landmarks = result.landmarks[0];

        drawingUtils.drawLandmarks(landmarks, { radius: 4 });
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);

        const shoulderIndex = hittingHand === "left" ? 11 : 12;
        const elbowIndex = hittingHand === "left" ? 13 : 14;
        const wristIndex = hittingHand === "left" ? 15 : 16;

        const rawShoulder = landmarks[shoulderIndex];
        const rawElbow = landmarks[elbowIndex];
        const rawWrist = landmarks[wristIndex];
        const rawNose = landmarks[0];

        smoothedPoints.shoulder = smoothPoint(smoothedPoints.shoulder, rawShoulder);
        smoothedPoints.elbow = smoothPoint(smoothedPoints.elbow, rawElbow);
        smoothedPoints.wrist = smoothPoint(smoothedPoints.wrist, rawWrist);
        smoothedPoints.nose = smoothPoint(smoothedPoints.nose, rawNose);

        const shoulder = smoothedPoints.shoulder;
        const elbow = smoothedPoints.elbow;
        const wrist = smoothedPoints.wrist;
        const nose = smoothedPoints.nose;

        const elbowAngle = angleBetweenThreePoints(shoulder, elbow, wrist);
        const extensionScore = getExtensionScore(elbowAngle);

        const wristY = wrist.y;
        const noseY = nose.y;
        const shoulderY = shoulder.y;

        const contactHeightScore = getContactHeightScore(wristY, noseY, shoulderY);
        const contactHeightLabel = getContactHeightLabel(contactHeightScore);

        const calibratedReach =
          hasCalibration && hasNetHeight
            ? estimateContactReachFromCalibration(
                wristY * canvas.height,
                hasStandingReach ? standingReachInches : null,
                netHeightInches
              )
            : null;

        const feedback = buildFeedback({
          elbowAngle,
          extensionScore,
          contactHeightScore,
          contactHeightLabel,
          calibratedReach,
          cameraAngle,
          repType,
          userNotes,
          hasStandingReach,
          hasNetHeight,
          hasCalibration
        });

        const armRaisedScore = (shoulderY - wristY) + ((elbowAngle || 0) / 500);

        if (!bestResult || armRaisedScore > bestResult.armRaisedScore) {
          bestResult = {
            elbowAngle,
            extensionScore,
            contactHeightScore,
            contactHeightLabel,
            calibratedReach,
            armRaisedScore,
            feedback
          };
        }
      }

      drawCalibrationMarkers();
    }

    requestAnimationFrame(processFrame);
  }

  if (heightInches > 0 && standingReachInches > 0 && standingReachInches < heightInches * 0.9) {
    output.textContent = "Your standing reach looks unusually low compared with your height. Double-check the numbers you entered.";
  } else {
    output.textContent = "Video loaded. Processing...";
  }

  video.play();
  processFrame();
}

canvas.addEventListener("click", (event) => {
  if (!calibrationMode) return;
  if (!video.videoWidth || !video.videoHeight) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;

  if (calibrationStep === 1) {
    netCalibration.netTop = { x, y };
    calibrationStep = 2;
    calibrationStatus.textContent = "Step 2: click the floor directly below the net.";
    return;
  }

  if (calibrationStep === 2) {
    netCalibration.floorPoint = { x, y };
    updateCalibrationScale();

    if (netCalibration.pixelsPerInch) {
      calibrationStatus.textContent = "Calibration set. Reach and net-margin estimates are now enabled.";
    } else {
      calibrationStatus.textContent = "Calibration failed. Reset and try again.";
    }

    calibrationMode = false;
    calibrationStep = 0;
  }
});

startCalibrationBtn.addEventListener("click", () => {
  const netHeightInches = feetInchesToInches(netFeetInput.value, netInchesInput.value);

  if (!video.src) {
    calibrationStatus.textContent = "Upload a video first.";
    return;
  }

  if (!netHeightInches) {
    calibrationStatus.textContent = "Enter net height first if you want calibration-based reach estimates.";
    return;
  }

  calibrationMode = true;
  calibrationStep = 1;
  netCalibration.netTop = null;
  netCalibration.floorPoint = null;
  netCalibration.pixelsPerInch = null;

  calibrationStatus.textContent = "Step 1: click the top of the net.";
});

resetCalibrationBtn.addEventListener("click", () => {
  resetCalibration();
});

videoInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  video.src = url;
  output.textContent = "Video loaded. Processing...";
  resetCalibration();

  video.onloadeddata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    analyzeVideo();
  };
});

await createPoseLandmarker();
output.textContent = "Model loaded. Upload a video to begin.";