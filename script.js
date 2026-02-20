const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let session;
const MODEL_PATH = './yolox_nano.onnx';
const INPUT_SIZE = 416; // Change to 640 if your model was exported at 640
const SCORE_THR = 0.3;
const IOU_THR = 0.45;

const LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

async function initAI() {
    status.innerText = "Waking up Smart Eyes...";
    try {
        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm'] // Safest for iPhone 13
        });
        status.innerText = "AI Online";
        startCamera();
    } catch (e) {
        status.innerText = "Error: Model file not found or incompatible.";
    }
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        runInference();
    };
}

async function runInference() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    
    const imgData = tCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const input = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    // NCHW Format & Normalization
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        input[i] = imgData.data[i * 4] / 255.0;
        input[i + INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 1] / 255.0;
        input[i + 2 * INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const feeds = { images: tensor }; // Check Netron if your input name is different
    const results = await session.run(feeds);
    
    // YOLOX Output Decoding
    const output = results.output.data; 
    const boxes = decodeYOLOX(output);
    renderBoxes(boxes);

    requestAnimationFrame(runInference);
}

function decodeYOLOX(data) {
    const detections = [];
    const rows = data.length / 85;

    for (let i = 0; i < rows; i++) {
        const row = i * 85;
        const conf = data[row + 4];
        if (conf < SCORE_THR) continue;

        let maxClassScore = 0;
        let classIdx = -1;
        for (let j = 0; j < 80; j++) {
            if (data[row + 5 + j] > maxClassScore) {
                maxClassScore = data[row + 5 + j];
                classIdx = j;
            }
        }

        if (maxClassScore * conf > SCORE_THR) {
            const w_factor = canvas.width / INPUT_SIZE;
            const h_factor = canvas.height / INPUT_SIZE;
            detections.push({
                bbox: [
                    (data[row] - data[row + 2] / 2) * w_factor,
                    (data[row + 1] - data[row + 3] / 2) * h_factor,
                    data[row + 2] * w_factor,
                    data[row + 3] * h_factor
                ],
                score: maxClassScore * conf,
                label: LABELS[classIdx]
            });
        }
    }
    return detections;
}

function renderBoxes(boxes) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    boxes.forEach(box => {
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 3;
        ctx.strokeRect(...box.bbox);
        
        ctx.fillStyle = "#00FF00";
        ctx.font = "18px Arial";
        ctx.fillText(`${box.label} (${Math.round(box.score * 100)}%)`, box.bbox[0], box.bbox[1] - 5);
        
        if (box.label === "person" && box.bbox[2] > canvas.width * 0.4) {
            speakOnce(`Careful, person very close.`);
        }
    });
}

let speaking = false;
function speakOnce(text) {
    if (!speaking) {
        speaking = true;
        const msg = new SpeechSynthesisUtterance(text);
        msg.onend = () => { setTimeout(() => speaking = false, 3000); };
        window.speechSynthesis.speak(msg);
    }
}

startBtn.addEventListener('click', () => {
    initAI();
    startBtn.style.display = 'none';
});
