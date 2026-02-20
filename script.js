import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const colorPreview = document.getElementById("color-preview");
const status = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

let detector;
let lastVideoTime = -1;
const FOCAL_LENGTH = 580; 
const KNOWN_WIDTHS = { "person": 0.45, "car": 1.8, "chair": 0.5, "bottle": 0.08 };

async function init() {
    status.innerText = "Initializing Color Brain...";
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    detector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU" 
        },
        scoreThreshold: 0.4,
        runningMode: "VIDEO"
    });
    status.innerText = "Vision Active";
    startCamera();
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 640, height: 480 } });
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
        renderDetections(result.detections);
    }
    window.requestAnimationFrame(predict);
}

// --- HSV COLOR LOGIC ---
function analyzeColor(x, y, w, h) {
    const sampleCanvas = document.createElement('canvas');
    sampleCanvas.width = 1; sampleCanvas.height = 1;
    const sCtx = sampleCanvas.getContext('2d');
    
    // Sample the center of the object
    sCtx.drawImage(video, x + w/4, y + h/4, w/2, h/2, 0, 0, 1, 1);
    const [r, g, b] = sCtx.getImageData(0, 0, 1, 1).data;

    let r_n = r/255, g_n = g/255, b_n = b/255;
    let max = Math.max(r_n, g_n, b_n), min = Math.min(r_n, g_n, b_n);
    let h_val, s_val = max === 0 ? 0 : (max - min) / max, v_val = max;
    let d = max - min;

    if (max === min) h_val = 0;
    else {
        if (max === r_n) h_val = (g_n - b_n) / d + (g_n < b_n ? 6 : 0);
        else if (max === g_n) h_val = (b_n - r_n) / d + 2;
        else h_val = (r_n - g_n) / d + 4;
        h_val /= 6;
    }

    const hue = h_val * 360;
    const sat = s_val * 100;
    const bright = v_val * 100;

    let colorName = "gray";
    let adjective = "";

    // Adjectives based on Saturation/Value
    if (sat > 70) adjective = "vibrant ";
    else if (sat < 20) adjective = "pale ";
    if (bright < 20) return "dark shadow";
    if (bright > 90 && sat < 10) return "bright white";

    if (hue < 15 || hue > 345) colorName = "red";
    else if (hue < 45) colorName = "orange";
    else if (hue < 75) colorName = "yellow";
    else if (hue < 160) colorName = "green";
    else if (hue < 260) colorName = "blue";
    else if (hue < 310) colorName = "purple";
    else colorName = "pink";

    return { name: adjective + colorName, rgb: `rgb(${r},${g},${b})`, sat: sat };
}

function renderDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(det => {
        const { originX, originY, width, height } = det.boundingBox;
        const label = det.categories[0].categoryName;
        const colorData = analyzeColor(originX, originY, width, height);
        
        const distance = ((KNOWN_WIDTHS[label] || 0.5) * FOCAL_LENGTH) / width;

        // Visual Overlay
        ctx.strokeStyle = colorData.rgb;
        ctx.lineWidth = 6;
        ctx.strokeRect(originX, originY, width, height);
        
        ctx.fillStyle = colorData.rgb;
        ctx.fillRect(originX, originY - 35, 200, 35);
        ctx.fillStyle = "white";
        ctx.fillText(`${colorData.name} ${label} - ${distance.toFixed(1)}m`, originX + 5, originY - 10);

        colorPreview.style.backgroundColor = colorData.rgb;

        // --- NEW: VIBRANCE FEEDBACK ---
        if (colorData.sat > 75 && navigator.vibrate) {
            navigator.vibrate(50); // Small "tick" for high-vibrance objects
        }

        // Narrator
        if (distance < 2) {
            triggerVoice(`A ${colorData.name} ${label} is nearby.`);
        }
    });
}

let speaking = false;
function triggerVoice(text) {
    if (speaking) return;
    speaking = true;
    const msg = new SpeechSynthesisUtterance(text);
    msg.onend = () => { setTimeout(() => speaking = false, 4000); };
    window.speechSynthesis.speak(msg);
}

startBtn.onclick = () => { init(); startBtn.style.display = "none"; };
