const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
let session;

// Configuration
const MODEL_PATH = './yolox_nano.onnx';
const CONF_THRESHOLD = 0.4;
const FOCAL_LENGTH = 500; // Average smartphone focal length in pixels

async function initAI() {
    status.innerText = "Loading AI Brain...";
    try {
        // Initialize ONNX Session with WebGPU
        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['webgpu']
        });
        status.innerText = "System Ready";
        startCamera();
    } catch (e) {
        status.innerText = "Error: Use a WebGPU-enabled browser (Chrome/Edge)";
        console.error(e);
    }
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" }, 
        audio: false 
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        runInference();
    };
}

async function runInference() {
    // 1. Preprocess: Resize frame to 640x640 (YOLOX standard)
    // 2. Inference: session.run()
    // 3. Post-process: Extract Bounding Boxes
    
    // Placeholder for detection results
    const detections = [ { label: 'Person', dist: 3.5, speed: 1.2 } ]; 

    detections.forEach(obj => {
        if (obj.dist < 2.0 && obj.speed > 0.5) {
            speak(`Warning: ${obj.label} approaching quickly. It is ${obj.dist} meters away. Move aside.`);
        }
    });

    requestAnimationFrame(runInference);
}

function speak(text) {
    if (!window.speechSynthesis.speaking) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.1; // Slightly faster for urgent alerts
        window.speechSynthesis.speak(utterance);
    }
}

startBtn.addEventListener('click', initAI);
