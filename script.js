import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const status = document.getElementById("status");

let detector;
let lastVideoTime = -1;
const FOCAL_LENGTH = 580; 
const KNOWN_WIDTHS = { "person": 0.5, "car": 1.8, "chair": 0.5, "bottle": 0.08 };

// 1. Voice Priming & Init
async function init() {
    // Prime the audio engine immediately on click
    const intro = new SpeechSynthesisUtterance("Smart Lens activating.");
    window.speechSynthesis.speak(intro);

    status.innerText = "Initializing Vision...";
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    
    detector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU" 
        },
        scoreThreshold: 0.4,
        runningMode: "VIDEO"
    });

    startCamera();
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment", width: 640, height: 480 } 
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        predict();
    };
}

async function predict() {
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const result = detector.detectForVideo(video, performance.now());
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        renderDetections(result.detections);
    }
    window.requestAnimationFrame(predict);
}

// 2. Color Analysis Logic
function getColor(x, y, w, h) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 1; tempCanvas.height = 1;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, x + w/2, y + h/2, 1, 1, 0, 0, 1, 1);
    const [r, g, b] = tCtx.getImageData(0, 0, 1, 1).data;
    
    // Convert to HSV for "Pretty" names
    let r_n = r/255, g_n = g/255, b_n = b/255;
    let max = Math.max(r_n, g_n, b_n), min = Math.min(r_n, g_n, b_n);
    let h_val, s_val = max === 0 ? 0 : (max - min) / max;
    let d = max - min;

    if (max === min) h_val = 0;
    else {
        if (max === r_n) h_val = (g_n - b_n) / d + (g_n < b_n ? 6 : 0);
        else if (max === g_n) h_val = (b_n - r_n) / d + 2;
        else h_val = (r_n - g_n) / d + 4;
        h_val /= 6;
    }

    const hue = h_val * 360;
    let name = "colorful";
    if (s_val < 0.15) name = "gray";
    else if (hue < 30 || hue > 330) name = "red";
    else if (hue < 90) name = "yellow";
    else if (hue < 150) name = "green";
    else if (hue < 240) name = "blue";
    else name = "purple";

    return { name, rgb: `rgb(${r},${g},${b})` };
}

// 3. Render and Narrate
function renderDetections(detections) {
    detections.forEach(det => {
        const { originX, originY, width, height } = det.boundingBox;
        const label = det.categories[0].categoryName;
        const color = getColor(originX, originY, width, height);
        const distance = ((KNOWN_WIDTHS[label] || 0.5) * FOCAL_LENGTH) / width;

        // VISUALS
        ctx.strokeStyle = color.rgb;
        ctx.lineWidth = 6;
        ctx.strokeRect(originX, originY, width, height);
        
        ctx.fillStyle = color.rgb;
        ctx.fillRect(originX, originY - 30, 180, 30);
        ctx.fillStyle = "white";
        ctx.font = "bold 16px sans-serif";
        ctx.fillText(`${color.name} ${label} ${distance.toFixed(1)}m`, originX + 5, originY - 10);

        // VOICE: Only describe if close or significant
        if (distance < 3) {
            triggerVoice(`A ${color.name} ${label} is ${distance.toFixed(1)} meters away.`);
        }
    });
}

let isSpeaking = false;
function triggerVoice(text) {
    if (isSpeaking) return;
    isSpeaking = true;
    const msg = new SpeechSynthesisUtterance(text);
    msg.onend = () => { setTimeout(() => isSpeaking = false, 3000); };
    window.speechSynthesis.speak(msg);
}

// Global click to restart audio if it hangs
window.onclick = () => {
    if (window.speechSynthesis.paused) window.speechSynthesis.resume();
};

startBtn.onclick = () => {
    init();
    startBtn.style.display = "none";
};
