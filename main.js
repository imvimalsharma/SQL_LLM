let session = null;
let tokenizer = null;
const VOCAB_SIZE = 4096;
const MAX_SEQ_LEN = 256;

// UI elements
const schemaInput = document.getElementById("schema");
const questionInput = document.getElementById("question");
const generateBtn = document.getElementById("generate-btn");
const outputPre = document.getElementById("output-sql");
const statusDiv = document.getElementById("status");
const progressContainer = document.getElementById("progress-container");
const progressBar = document.getElementById("progress-bar");

async function logStatus(message, isError = false) {
    statusDiv.innerHTML = message;
    if (isError) {
        statusDiv.className = "status-error";
    } else {
        statusDiv.className = "status-info";
    }
    console.log(message);
}

// 1. Initialize ONNX Runtime and Tokenizer
async function init() {
    try {
        logStatus("Initializing custom BPE tokenizer...");
        tokenizer = new window.BPETokenizer();
        
        // Fetch tokenizer config
        const response = await fetch("data/tokenizer.json");
        if (!response.ok) {
            throw new Error("Failed to load data/tokenizer.json. Make sure training completed.");
        }
        const tokenizerData = await response.json();
        tokenizer.load(tokenizerData);
        logStatus("Tokenizer loaded. Now loading 8.5MB quantized LLaMA-style ONNX model (fast load)...");
        
        // Setup progress bar simulation for loading model
        progressContainer.style.display = "block";
        progressBar.style.width = "30%";
        
        // Configure ONNX Runtime to use WASM
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        
        // Create Inference Session
        session = await ort.InferenceSession.create("./model_quant.onnx", {
            executionProviders: ["wasm"]
        });
        
        progressBar.style.width = "100%";
        setTimeout(() => {
            progressContainer.style.display = "none";
        }, 500);
        
        logStatus("⚡ Model loaded successfully! Ready to generate SQL.");
        generateBtn.disabled = false;
    } catch (err) {
        logStatus(`Initialization failed: ${err.message}`, true);
        progressBar.style.backgroundColor = "#ff4d4d";
    }
}

// 2. Autoregressive SQL Generation Loop
async function generateSQL() {
    const schema = schemaInput.value.trim();
    const question = questionInput.value.trim();
    
    if (!schema || !question) {
        alert("Please enter both a database schema and a question.");
        return;
    }
    
    generateBtn.disabled = true;
    outputPre.textContent = "";
    logStatus("Generating SQL query...");
    
    // Construct standard instruction prompt
    const prompt = `[INST] Schema: ${schema} | Question: ${question} [/INST] SQL:`;
    
    // Encode prompt into token IDs
    let tokenIds = tokenizer.encode(prompt);
    const startLen = tokenIds.length;
    
    let generatedSQL = "";
    
    try {
        // Generate up to 128 new tokens
        for (let step = 0; step < 128; step++) {
            // Context window cropping to max_seq_len (256)
            let contextIds = tokenIds;
            if (tokenIds.length > MAX_SEQ_LEN) {
                contextIds = tokenIds.slice(-MAX_SEQ_LEN);
            }
            
            // Convert to BigInt64Array for ONNX Runtime int64 inputs
            const inputArray = new BigInt64Array(contextIds.map(BigInt));
            
            // Create inputs tensor of shape [1, sequence_length]
            const inputTensor = new ort.Tensor("int64", inputArray, [1, contextIds.length]);
            
            // Run model inference
            const outputs = await session.run({ idx: inputTensor });
            const logits = outputs.logits; // shape [1, seq_len, vocab_size]
            
            // Extract the logits of the last token in sequence
            const seqLen = contextIds.length;
            const offset = (seqLen - 1) * VOCAB_SIZE;
            const lastLogits = logits.data.subarray(offset, offset + VOCAB_SIZE);
            
            // Deterministic Greedy Search (Argmax) for highest accuracy SQL
            let maxVal = -Infinity;
            let nextTokenId = 0;
            for (let i = 0; i < VOCAB_SIZE; i++) {
                if (lastLogits[i] > maxVal) {
                    maxVal = lastLogits[i];
                    nextTokenId = i;
                }
            }
            
            // End of text token generated (<|endoftext|> is token 256)
            if (nextTokenId === 256) {
                break;
            }
            
            // Append token to input sequence for next iteration
            tokenIds.push(nextTokenId);
            
            // Decode and display current token (streaming output)
            const tokenStr = tokenizer.decode([nextTokenId]);
            generatedSQL += tokenStr;
            outputPre.textContent = generatedSQL;
            
            // Brief yield to keep UI responsive
            await new Promise(resolve => setTimeout(resolve, 5));
        }
        
        logStatus("⚡ SQL Generation complete!");
    } catch (err) {
        logStatus(`Generation error: ${err.message}`, true);
    } finally {
        generateBtn.disabled = false;
    }
}

// Bind generate button event click
generateBtn.addEventListener("click", generateSQL);

// Run init on window load
window.addEventListener("load", init);
