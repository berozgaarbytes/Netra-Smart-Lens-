import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

let detector;
let lastVideoTime = -1;
let lastSceneDescriptionTime = 0;
let currentDetections = [];

const REAL_WIDTHS = { "person": 0.5, "car": 1.8, "bicycle": 0.6, "chair": 0.5, "bottle": 0.08 };

// --- AUDIO PRIMING FOR IOS ---
function primeAudio() {
    // Speak a silent or short message to "unlock" the audio channel in Safari
    const utter = new SpeechSynthesisUtterance("Voice engine activated.");
    utter.volume = 0.1; 
    window.speechSynthesis.speak(utter);
}

async function init() {
    primeAudio(); // This must be inside the click event!
    status.innerText = "Loading Vision...";
    
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

    status.innerText = "Smart Lens Running";
    startCamera();
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" } 
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
        currentDetections = result.detections;
        processScene();
    }
    window.requestAnimationFrame(predict);
}

function processScene() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let foundObjects = [];

    currentDetections.forEach(det => {
        const { originX, width } = det.boundingBox;
        const label = det.categories[0].categoryName;
        const distance = ((REAL_WIDTHS[label] || 0.5) * 550) / width;
        
        foundObjects.push(label);

        // Immediate Collision Warning (Higher Priority)
        if (distance < 1.5) {
            speak(`Warning: ${label} very close, ${distance.toFixed(1)} meters.`);
        }
    });

    // Scene Narrator (Describes surrounding every 8 seconds)
    const now = Date.now();
    if (now - lastSceneDescriptionTime > 8000 && foundObjects.length > 0) {
        const uniqueObjects = [...new Set(foundObjects)];
        speak(`In front of you, I see: ${uniqueObjects.join(", ")}.`);
        lastSceneDescriptionTime = now;
    }
}

let isSpeaking = false;
function speak(text) {
    if (isSpeaking) return;
    isSpeaking = true;
    
    const msg = new SpeechSynthesisUtterance(text);
    msg.onend = () => { isSpeaking = false; };
    window.speechSynthesis.speak(msg);
}

startBtn.onclick = () => {
    init();
    startBtn.style.display = "none";
};
