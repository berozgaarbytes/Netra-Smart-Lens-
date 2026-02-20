const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let session;
const MODEL_PATH = './yolox_nano.onnx';
const CONF_THRESHOLD = 0.35; // Sensitivity (0.0 to 1.0)
const INPUT_SIZE = 416;      // YOLOX-Nano default is 416x416

// COCO Labels (Simplified for the blind)
const LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

async function initAI() {
    status.innerText = "Initializing AI...";
    try {
        // iOS/Safari fix: Using WASM for maximum compatibility
        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm']
        });
        status.innerText = "System Online";
        startCamera();
    } catch (e) {
        status.innerText = "Error loading model. Check console.";
        console.error(e);
    }
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment", width: 640, height: 640 }, 
        audio: false 
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        runInference();
    };
}

async function runInference() {
    if (!session) return;

    // 1. Prepare Input Canvas
    const offscreen = document.createElement('canvas');
    offscreen.width = INPUT_SIZE;
    offscreen.height = INPUT_SIZE;
    const osCtx = offscreen.getContext('2d');
    osCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    
    // 2. Preprocess: Normalize to [0, 1] and NCHW format
    const imgData = osCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const floatData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        floatData[i] = imgData.data[i * 4] / 255.0;           // R
        floatData[i + INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 1] / 255.0; // G
        floatData[i + 2 * INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    
    // 3. Inference
    const results = await session.run({ images: tensor }); // 'images' is standard for YOLOX ONNX
    const output = results.output.data; // Shape [1, 3549, 85] for Nano 416

    // 4. Post-Process
    processDetections(output);

    setTimeout(runInference, 100); // Throttled for battery life
}

function processDetections(data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const numDetections = data.length / 85;

    for (let i = 0; i < numDetections; i++) {
        const offset = i * 85;
        const objConf = data[offset + 4]; // Objectness Score

        if (objConf > CONF_THRESHOLD) {
            // Get Class
            let maxScore = 0;
            let classIdx = -1;
            for (let j = 0; j < 80; j++) {
                if (data[offset + 5 + j] > maxScore) {
                    maxScore = data[offset + 5 + j];
                    classIdx = j;
                }
            }

            if (maxScore * objConf > CONF_THRESHOLD) {
                // Coordinates (cx, cy, w, h)
                const cx = data[offset] * (canvas.width / INPUT_SIZE);
                const cy = data[offset + 1] * (canvas.height / INPUT_SIZE);
                const w = data[offset + 2] * (canvas.width / INPUT_SIZE);
                const h = data[offset + 3] * (canvas.height / INPUT_SIZE);

                // Draw Box
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 2;
                ctx.strokeRect(cx - w/2, cy - h/2, w, h);

                // Simple Distance Logic: Based on known object widths
                const distance = (0.5 * canvas.width) / w; // Approximation
                
                if (classIdx === 0) { // If Person
                    speakAlert(`Person detected, ${distance.toFixed(1)} meters away.`);
                }
            }
        }
    }
}

let lastSpoke = 0;
function speakAlert(text) {
    const now = Date.now();
    if (now - lastSpoke > 4000) { // Limit speech to once every 4 seconds
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
        lastSpoke = now;
    }
}

startBtn.addEventListener('click', () => {
    initAI();
    startBtn.style.display = 'none'; // Hide button after start
});
