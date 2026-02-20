import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
let objectDetector;

async function initialize() {
  status.innerText = "Loading Stable Brain...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  
  objectDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
      delegate: "GPU" // iOS 26 handles this delegate well
    },
    scoreThreshold: 0.4,
    runningMode: "VIDEO"
  });

  status.innerText = "Lens Active";
  startCamera();
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ 
    video: { facingMode: "environment" } 
  });
  video.srcObject = stream;
  video.addEventListener("loadeddata", predict);
}

let lastVideoTime = -1;
async function predict() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const detections = objectDetector.detectForVideo(video, performance.now());
    displayDetections(detections);
  }
  window.requestAnimationFrame(predict);
}

function displayDetections(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  result.detections.forEach((detection) => {
    const { originX, originY, width, height } = detection.boundingBox;
    const label = detection.categories[0].categoryName;
    
    // Draw Box
    ctx.strokeStyle = "#00E676";
    ctx.lineWidth = 4;
    ctx.strokeRect(originX, originY, width, height);

    // Label
    ctx.fillStyle = "#00E676";
    ctx.font = "bold 20px Arial";
    ctx.fillText(label, originX, originY > 20 ? originY - 10 : 20);

    // Audio Alert
    if (label === "person") speakOnce("Person detected");
  });
}

let speaking = false;
function speakOnce(text) {
  if (!speaking) {
    speaking = true;
    const msg = new SpeechSynthesisUtterance(text);
    msg.onend = () => setTimeout(() => speaking = false, 3000);
    window.speechSynthesis.speak(msg);
  }
}

document.getElementById("startBtn").onclick = initialize;
