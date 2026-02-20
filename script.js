const video = document.getElementById('webcam');
const status = document.getElementById('status');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let session;
const MODEL_PATH = './yolox_nano.onnx';
const INPUT_SIZE = 416; // 416 is much safer for iPhone memory than 640

async function initAI() {
    status.innerText = "Starting Ultra-Safe Mode...";
    try {
        // LOCK threads to 1 and disable SIMD to prevent Safari memory spikes
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = false; 

        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm'],
            enableCpuMemArena: false // Prevents the browser from grabbing too much RAM at once
        });
        
        status.innerText = "System Active";
        startCamera();
    } catch (e) {
        status.innerText = "System Failure: " + e.message;
    }
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment", width: 416, height: 416 } // Low res camera to save memory
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

    // Use a small temporary canvas for resizing
    const off = document.createElement('canvas');
    off.width = INPUT_SIZE;
    off.height = INPUT_SIZE;
    const oCtx = off.getContext('2d');
    oCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    
    const imgData = oCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const floatData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    // Normalize and Reorder
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        floatData[i] = imgData.data[i * 4] / 255.0;           
        floatData[i + INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 1] / 255.0;
        floatData[i + 2 * INPUT_SIZE * INPUT_SIZE] = imgData.data[i * 4 + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    
    try {
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;
        const results = await session.run(feeds);
        
        // Let's just draw ONE box in the middle to confirm the model is alive
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 5;
        ctx.strokeRect(50, 50, 100, 100); // If you see this red box, the loop is running
        
        status.innerText = "Processing Frames...";
    } catch (err) {
        status.innerText = "Inference Error";
    }

    // Delay 500ms (2 frames per second) - This is to prevent the crash!
    setTimeout(runInference, 500); 
}

document.getElementById('startBtn').onclick = () => {
    initAI();
    document.getElementById('startBtn').style.display = 'none';
};
