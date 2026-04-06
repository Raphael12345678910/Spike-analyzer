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
const contactValue = document.getElementById("contactValue");

let poseLandmarker;
let drawingUtils;
let bestResult = null;

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

function angleBetweenThreePoints(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const magCB = Math.sqrt(cb.x * cb.x + cb.y * cb.y);

  if (magAB === 0 || magCB === 0) return null;

  let cosine = dot / (magAB * magCB);
  cosine = Math.max(-1, Math.min(1, cosine));

  const angle = Math.acos(cosine) * (180 / Math.PI);
  return angle;
}

function getFeedback(elbowAngle, contactLabel) {
  const tips = [];

  if (elbowAngle !== null) {
    if (elbowAngle < 145) {
      tips.push(
        "Your hitting arm stays bent near contact, which may reduce reach and power. Try extending more fully at contact."
      );
    } else {
      tips.push(
        "Your arm extension looks fairly strong at contact."
      );
    }
  }

  if (contactLabel === "Below head") {
    tips.push(
      "Your contact point is below head level, which likely reduces your hitting angle. Try reaching higher at contact."
    );
  } else if (contactLabel === "Near head level") {
    tips.push(
      "Your contact point is close to head level. You may benefit from reaching slightly higher."
    );
  } else {
    tips.push(
      "Your contact point is above head level, which is generally good."
    );
  }

  return tips.join("\n");
}

async function analyzeVideo() {
  if (!poseLandmarker) {
    output.textContent = "Pose model is still loading.";
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  let lastTime = -1;
  bestResult = null;

  async function processFrame() {
    if (video.paused || video.ended) {
      if (bestResult) {
        angleValue.textContent = bestResult.elbowAngle
          ? `${bestResult.elbowAngle.toFixed(1)}°`
          : "--";

        contactValue.textContent = bestResult.contactLabel || "--";

        output.textContent =
          `Main feedback:\n${bestResult.feedback}`;
      } else {
        angleValue.textContent = "--";
        contactValue.textContent = "--";
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
        drawingUtils.drawConnectors(
          landmarks,
          PoseLandmarker.POSE_CONNECTIONS
        );

        const rightShoulder = landmarks[12];
        const rightElbow = landmarks[14];
        const rightWrist = landmarks[16];
        const nose = landmarks[0];

        const elbowAngle = angleBetweenThreePoints(
          rightShoulder,
          rightElbow,
          rightWrist
        );

        const wristY = rightWrist.y;
        const noseY = nose.y;

        let contactLabel = "Unknown";

        if (wristY > noseY + 0.05) {
          contactLabel = "Below head";
        } else if (wristY > noseY - 0.03) {
          contactLabel = "Near head level";
        } else {
          contactLabel = "Above head";
        }

        const feedback = getFeedback(elbowAngle, contactLabel);

        if (!bestResult || wristY < bestResult.wristY) {
          bestResult = {
            elbowAngle,
            wristY,
            contactLabel,
            feedback
          };
        }
      }
    }

    requestAnimationFrame(processFrame);
  }

  video.play();
  processFrame();
}

videoInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  video.src = url;
  output.textContent = "Video loaded. Processing...";

  video.onloadeddata = () => {
    analyzeVideo();
  };
});

await createPoseLandmarker();
output.textContent = "Model loaded. Upload a video to begin.";