const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let session;
const MODEL_PATH = './yolox_nano.onnx';
const INPUT_SIZE = 416; // Use 416 for Nano/Tiny models to save memory
const SCORE_THR = 0.4;
const IOU_THR = 0.4;

const LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

async function initAI() {
    status.innerText = "Starting Safe-Memory AI...";
    try {
        // Critical: Prevents iOS from crashing by limiting CPU threads to 1
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = false; 

        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm'],
            enableCpuMemArena: false // Disabling arena prevents large memory pre-allocation
        });
        
        status.innerText = "Lens Active";
        startCamera();
    } catch (e) {
        status.innerText = "Error: Use a lower-res model or check path.";
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

    // Create a 416x416 frame for the AI
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    
    const imgData = tCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const input = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    // Normalization and channel reordering
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        input[i] = imgData.data[i * 4] / 255.0;           // R
        input[i + INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 1] / 255.0; // G
        input[i + 2 * INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    
    // Check 'images' vs 'input' node name in Netron if this line fails
    const results = await session.run({ images: tensor }); 
    const output = results.output.data; 

    const detections = processYOLOX(output);
    drawVisuals(detections);

    // Run every 200ms to keep iPhone cool and stable
    setTimeout(runInference, 200); 
}

function processYOLOX(data) {
    const boxes = [];
    const rows = data.length / 85;

    for (let i = 0; i < rows; i++) {
        const row = i * 85;
        const objConf = data[row + 4];
        if (objConf < SCORE_THR) continue;

        let classScore = 0;
        let classIdx = -1;
        for (let j = 0; j < 80; j++) {
            if (data[row + 5 + j] > classScore) {
                classScore = data[row + 5 + j];
                classIdx = j;
            }
        }

        const finalScore = objConf * classScore;
        if (finalScore > SCORE_THR) {
            const w_scale = canvas.width / INPUT_SIZE;
            const h_scale = canvas.height / INPUT_SIZE;
            boxes.push({
                bbox: [
                    (data[row] - data[row + 2] / 2) * w_scale,
                    (data[row + 1] - data[row + 3] / 2) * h_scale,
                    data[row + 2] * w_scale,
                    data[row + 3] * h_scale
                ],
                score: finalScore,
                label: LABELS[classIdx]
            });
        }
    }
    return boxes;
}

function drawVisuals(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    detections.forEach(d => {
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 4;
        ctx.strokeRect(...d.bbox);
        
        ctx.fillStyle = "#00FF00";
        ctx.font = "bold 20px Arial";
        ctx.fillText(`${d.label}`, d.bbox[0], d.bbox[1] - 10);
        
        // Simple Audio Logic for Collision
        if (d.bbox[2] > canvas.width * 0.5) { // If object takes up half the screen
            speak(`Object close: ${d.label}`);
        }
    });
}

let speaking = false;
function speak(text) {
    if (!speaking) {
        speaking = true;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => { setTimeout(() => speaking = false, 3000); };
        window.speechSynthesis.speak(utterance);
    }
}

startBtn.addEventListener('click', () => {
    initAI();
    startBtn.style.display = 'none';
});
