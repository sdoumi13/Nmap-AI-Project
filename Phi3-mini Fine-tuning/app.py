import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =====================
# Config
# =====================
BASE_MODEL_PATH = r"C:\models\phi3_mini"
ADAPTER_PATH = "./phi3-nmap-results"

# =====================
# Quantization
# =====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# =====================
# Load Model (ONCE)
# =====================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=False,
    local_files_only=True
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# =====================
# FastAPI App
# =====================
app = FastAPI(title="Nmap Command Generator API")

class PromptRequest(BaseModel):
    prompt: str

# =====================
# Inference Function
# =====================
def generate_nmap(prompt: str) -> str:
    system_msg = "Output a precise nmap command for the given task."
    full_prompt = f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{prompt} [/INST]"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("[/INST]")[-1].strip()

# =====================
# API Endpoint
# =====================
@app.post("/generate")
def generate(request: PromptRequest):
    result = generate_nmap(request.prompt)
    return {"nmap_command": result}

# =====================
# Health Check
# =====================
@app.get("/")
def health():
    return {"status": "ok"}
