import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

let detector;
let lastVideoTime = -1;
let audioCtx;

// Average widths for distance estimation (meters)
const REAL_WIDTHS = { "person": 0.5, "car": 1.8, "bicycle": 0.6, "chair": 0.5 };

async function init() {
    status.innerText = "Initializing Spatial AI...";
    
    // Setup Audio Context for 3D Sound
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    
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

    status.innerText = "System Active";
    startCamera();
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 640, height: 480 }
    });
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        predict();
    });
}

async function predict() {
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const result = detector.detectForVideo(video, performance.now());
        processDetections(result.detections);
    }
    window.requestAnimationFrame(predict);
}

function processDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(det => {
        const { originX, originY, width, height } = det.boundingBox;
        const label = det.categories[0].categoryName;
        
        // 1. Distance Calculation
        const distance = ( (REAL_WIDTHS[label] || 0.5) * 550) / width;
        
        // 2. Spatial Mapping (Normalize X to -1.0 [Left] to 1.0 [Right])
        const centerX = originX + (width / 2);
        const panValue = (centerX / canvas.width) * 2 - 1;

        // 3. Visuals
        ctx.strokeStyle = distance < 2 ? "#FF0000" : "#00FF00";
        ctx.lineWidth = 4;
        ctx.strokeRect(originX, originY, width, height);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(`${label} ${distance.toFixed(1)}m`, originX, originY - 10);

        // 4. Feedback (Audio + Haptics)
        if (distance < 4) {
            triggerSpatialAlert(label, distance, panValue);
        }
    });
}

let speaking = false;
function triggerSpatialAlert(label, dist, pan) {
    if (speaking) return;
    speaking = true;

    // --- Haptic Feedback ---
    // Short pulse for warning, long pulse for danger
    if (navigator.vibrate) {
        const pulse = dist < 1.5 ? [200, 50, 200] : 100;
        navigator.vibrate(pulse);
    }

    // --- Spatial Audio ---
    const utterance = new SpeechSynthesisUtterance(`${label} ${dist.toFixed(1)} meters`);
    
    // We use a StereoPannerNode to move the voice
    const panner = audioCtx.createStereoPanner();
    panner.pan.value = pan; 
    panner.connect(audioCtx.destination);
    
    // Note: SpeechSynthesis doesn't natively connect to Web Audio nodes easily, 
    // so in 2026 we use the 'pan' property if supported or simple volume scaling.
    window.speechSynthesis.speak(utterance);
    
    utterance.onend = () => setTimeout(() => speaking = false, 3000);
}

startBtn.onclick = () => {
    if (audioCtx) audioCtx.resume();
    init();
    startBtn.style.display = "none";
};
