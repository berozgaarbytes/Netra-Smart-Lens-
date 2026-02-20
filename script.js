const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let session;
const MODEL_PATH = './yolox_nano.onnx';
const INPUT_SIZE = 416; // Most YOLOX Nano use 416. Try 640 if this fails.
const CONF_THRESHOLD = 0.3;

const LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

async function initAI() {
    status.innerText = "Starting Lens...";
    try {
        // Force single-thread WASM for iPhone stability
        ort.env.wasm.numThreads = 1;
        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm']
        });
        status.innerText = "System Ready. Point at something!";
        startCamera();
    } catch (e) {
        status.innerText = "Error: " + e.message;
        console.error(e);
    }
}

async function startCamera() {
    try {
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
    } catch (err) {
        status.innerText = "Camera Error: " + err.message;
    }
}

async function runInference() {
    if (!session) return;

    // 1. Prepare 416x416 Input
    const offscreen = document.createElement('canvas');
    offscreen.width = INPUT_SIZE;
    offscreen.height = INPUT_SIZE;
    const osCtx = offscreen.getContext('2d');
    osCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    
    const imgData = osCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const floatData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    // Normalization (Divide by 255) and RGB to Channel-First (NCHW)
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        floatData[i] = imgData.data[i * 4] / 255.0;           // R
        floatData[i + INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 1] / 255.0; // G
        floatData[i + 2 * INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    
    try {
        // Try 'images' as input name, fallback to first input name if it fails
        const inputName = session.inputNames[0];
        const feeds = {};
        feeds[inputName] = tensor;
        
        const results = await session.run(feeds);
        const outputName = session.outputNames[0];
        const output = results[outputName].data;

        processDetections(output);
    } catch (err) {
        console.error("Inference failed:", err);
    }

    setTimeout(runInference, 150); // Small delay to prevent iPhone overheating
}

function processDetections(data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const rows = data.length / 85;
    let foundCount = 0;

    for (let i = 0; i < rows; i++) {
        const offset = i * 85;
        const objConf = data[offset + 4];

        if (objConf > CONF_THRESHOLD) {
            let maxClassScore = 0;
            let classIdx = -1;
            for (let j = 0; j < 80; j++) {
                if (data[offset + 5 + j] > maxClassScore) {
                    maxClassScore = data[offset + 5 + j];
                    classIdx = j;
                }
            }

            if (maxClassScore * objConf > CONF_THRESHOLD) {
                foundCount++;
                const w_factor = canvas.width / INPUT_SIZE;
                const h_factor = canvas.height / INPUT_SIZE;
                
                // YOLOX format: [cx, cy, w, h]
                const w = data[offset + 2] * w_factor;
                const h = data[offset + 3] * h_factor;
                const x = (data[offset] * w_factor) - (w / 2);
                const y = (data[offset + 1] * h_factor) - (h / 2);

                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 5;
                ctx.strokeRect(x, y, w, h);
                
                ctx.fillStyle = "#00FF00";
                ctx.font = "bold 24px Arial";
                ctx.fillText(LABELS[classIdx], x, y > 30 ? y - 10 : y + 30);
                
                if (LABELS[classIdx] === "person") {
                    triggerVoice(`Person detected`);
                }
            }
        }
    }
    status.innerText = `Active - Objects Found: ${foundCount}`;
}

let lastSpeechTime = 0;
function triggerVoice(text) {
    const now = Date.now();
    if (now - lastSpeechTime > 5000) { // Limit to every 5 seconds
        const msg = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(msg);
        lastSpeechTime = now;
    }
}

startBtn.addEventListener('click', () => {
    initAI();
    startBtn.style.display = 'none';
});
