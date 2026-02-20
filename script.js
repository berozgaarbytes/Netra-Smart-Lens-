const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let session;
const MODEL_PATH = './yolox_nano.onnx';
const INPUT_SIZE = 416; // Standard YOLOX-Nano size
const SCORE_THRESHOLD = 0.3;

// COCO Labels
const LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

async function initAI() {
    status.innerText = "Loading AI Engine...";
    try {
        // iOS Stability Fix
        ort.env.wasm.numThreads = 1;
        
        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm']
        });
        
        status.innerText = "System Live. Point at a person.";
        startCamera();
    } catch (e) {
        status.innerText = "Error: Could not load model. Check filename.";
        console.error(e);
    }
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment", width: 640, height: 640 } 
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

    // 1. Create Input Tensor
    const off = document.createElement('canvas');
    off.width = INPUT_SIZE;
    off.height = INPUT_SIZE;
    const oCtx = off.getContext('2d');
    oCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    
    const imgData = oCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const floatData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    // NCHW Normalization: Crucial for YOLOX
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        floatData[i] = imgData.data[i * 4] / 255.0;           // R
        floatData[i + INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 1] / 255.0; // G
        floatData[i + 2 * INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    
    try {
        // Auto-detect input/output names
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]].data;

        const detections = parseYOLOX(output);
        draw(detections);
    } catch (err) {
        console.warn("Inference hiccup:", err);
    }

    // Delay 200ms to prevent crashing/overheating
    setTimeout(runInference, 200);
}

function parseYOLOX(data) {
    const boxes = [];
    // YOLOX Nano typically outputs 3549 rows for 416x416
    const rows = data.length / 85; 

    for (let i = 0; i < rows; i++) {
        const row = i * 85;
        const objConf = data[row + 4];
        if (objConf < SCORE_THRESHOLD) continue;

        let clsScore = 0;
        let clsIdx = -1;
        for (let j = 0; j < 80; j++) {
            if (data[row + 5 + j] > clsScore) {
                clsScore = data[row + 5 + j];
                clsIdx = j;
            }
        }

        const score = objConf * clsScore;
        if (score > SCORE_THRESHOLD) {
            const w_scale = canvas.width / INPUT_SIZE;
            const h_scale = canvas.height / INPUT_SIZE;
            
            // YOLOX format is [cx, cy, w, h]
            const w = data[row + 2] * w_scale;
            const h = data[row + 3] * h_scale;
            const x = (data[row] * w_scale) - (w / 2);
            const y = (data[row + 1] * h_scale) - (h / 2);

            boxes.push({ box: [x, y, w, h], label: LABELS[clsIdx], score });
        }
    }
    return boxes;
}

function draw(dets) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    status.innerText = `Active - Objects Found: ${dets.length}`;

    dets.forEach(d => {
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 4;
        ctx.strokeRect(...d.box);
        
        ctx.fillStyle = "#00FF00";
        ctx.font = "bold 20px Arial";
        ctx.fillText(`${d.label}`, d.box[0], d.box[1] > 20 ? d.box[1] - 10 : 30);
        
        if (d.label === "person") voice(`Person nearby`);
    });
}

let lastSpeach = 0;
function voice(t) {
    if (Date.now() - lastSpeach > 5000) {
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(t));
        lastSpeach = Date.now();
    }
}

startBtn.addEventListener('click', () => {
    initAI();
    startBtn.style.display = 'none';
});
