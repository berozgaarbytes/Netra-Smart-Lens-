import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const colorPreview = document.getElementById("color-preview");
const startBtn = document.getElementById("startBtn");

let detector;
let lastVideoTime = -1;
let preferredVoice = null;

// Initialize Voice Selection
function loadVoices() {
    const voices = window.speechSynthesis.getVoices();
    // Try to find a "Premium" or "Natural" sounding voice
    preferredVoice = voices.find(v => v.name.includes("Samantha") || v.name.includes("Google US English")) || voices[0];
}
window.speechSynthesis.onvoiceschanged = loadVoices;

async function init() {
    loadVoices();
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
        // MATCH CANVAS TO VIDEO PIXELS
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        predict();
    };
}

async function predict() {
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const result = detector.detectForVideo(video, performance.now());
        
        // CLEAR CANVAS BEFORE DRAWING
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        renderDetections(result.detections);
    }
    window.requestAnimationFrame(predict);
}

function renderDetections(detections) {
    detections.forEach(det => {
        const { originX, originY, width, height } = det.boundingBox;
        const label = det.categories[0].categoryName;
        
        // Sampling color
        const colorData = analyzeColor(originX, originY, width, height);

        // DRAWING ON CANVAS
        ctx.strokeStyle = colorData.rgb;
        ctx.lineWidth = 8; // Extra thick for visibility
        ctx.strokeRect(originX, originY, width, height);
        
        ctx.fillStyle = colorData.rgb;
        ctx.font = "bold 24px Arial";
        ctx.fillText(`${colorData.name} ${label}`, originX, originY > 30 ? originY - 10 : 40);

        colorPreview.style.backgroundColor = colorData.rgb;
        
        // Speak if close
        if (width > canvas.width * 0.4) {
            triggerVoice(`A beautiful ${colorData.name} ${label} is right here.`);
        }
    });
}

function analyzeColor(x, y, w, h) {
    // We use a small internal sample to get Hue/Sat/Val
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 1; tempCanvas.height = 1;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, x + w/2, y + h/2, 1, 1, 0, 0, 1, 1);
    const [r, g, b] = tCtx.getImageData(0, 0, 1, 1).data;
    
    // Simple color name logic (Simplified for space)
    const rgbStr = `rgb(${r},${g},${b})`;
    return { name: "colorful", rgb: rgbStr }; 
}

let speaking = false;
function triggerVoice(text) {
    if (speaking) return;
    speaking = true;
    const msg = new SpeechSynthesisUtterance(text);
    msg.voice = preferredVoice; // USE THE PREFERRED VOICE
    msg.rate = 0.9;
    msg.pitch = 1.1; // Make it a bit more friendly/high-pitched
    msg.onend = () => { setTimeout(() => speaking = false, 4000); };
    window.speechSynthesis.speak(msg);
}

startBtn.onclick = () => {
    init();
    startBtn.style.display = "none";
};
