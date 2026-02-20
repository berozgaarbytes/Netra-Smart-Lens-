import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

let detector;
let lastVideoTime = -1;

// Real-world widths in meters for distance estimation
const REAL_WIDTHS = {
    "person": 0.5,
    "car": 1.8,
    "bicycle": 0.6,
    "dog": 0.3,
    "chair": 0.5,
    "bottle": 0.08
};

// 1. Initialize the Detector
async function init() {
    status.innerText = "Waking up AI...";
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    
    detector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU" 
        },
        scoreThreshold: 0.4,
        runningMode: "VIDEO"
    });

    status.innerText = "Lens Active";
    startCamera();
}

// 2. Start Camera
async function startCamera() {
    const constraints = {
        video: { facingMode: "environment", width: 640, height: 480 }
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.addEventListener("loadeddata", predict);
}

// 3. Predict & Draw
async function predict() {
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const result = detector.detectForVideo(video, performance.now());
        processAndDraw(result.detections);
    }
    window.requestAnimationFrame(predict);
}

// 4. Distance Calculation & Visualization
function processAndDraw(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    detections.forEach(det => {
        const { originX, originY, width, height } = det.boundingBox;
        const label = det.categories[0].categoryName;
        
        // --- Distance Estimation Math ---
        // Formula: Distance = (RealWidth * FocalLength) / PixelWidth
        // 500 is a common approximate focal length for mobile browsers
        const realWidth = REAL_WIDTHS[label] || 0.5; 
        const distance = (realWidth * 550) / (width * (640 / canvas.width));
        const distFixed = distance.toFixed(1);

        // Draw Styling
        const color = distance < 2 ? "#FF3D00" : "#00E676"; // Red if < 2m
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(originX, originY, width, height);

        ctx.fillStyle = color;
        ctx.font = "bold 20px Arial";
        ctx.fillText(`${label.toUpperCase()} - ${distFixed}m`, originX, originY > 20 ? originY - 10 : 20);

        // Voice Feedback Logic
        if (distance < 3) {
            handleVoiceAlert(label, distFixed);
        }
    });
}

let speaking = false;
function handleVoiceAlert(label, dist) {
    if (!speaking) {
        speaking = true;
        const urgency = dist < 1.5 ? "Urgent: " : "";
        const text = `${urgency}${label} is ${dist} meters away.`;
        
        const msg = new SpeechSynthesisUtterance(text);
        msg.onend = () => setTimeout(() => speaking = false, 3000); // 3s cool down
        window.speechSynthesis.speak(msg);
    }
}

startBtn.onclick = () => {
    init();
    startBtn.style.display = "none";
};
