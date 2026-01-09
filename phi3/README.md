# Nmap Command Generator API

An intelligent FastAPI-based service that generates precise nmap commands using a fine-tuned Phi-3 model with LoRA adapters. Built with quantization for efficient inference.

## Features

- **AI-Powered Command Generation**: Uses a fine-tuned Phi-3 mini model to generate nmap commands
- **LoRA Adapters**: Efficient fine-tuning using Low-Rank Adaptation for task-specific customization
- **4-bit Quantization**: Reduces model size and memory requirements while maintaining quality
- **FastAPI Backend**: Modern async REST API for easy integration
- **CUDA Support**: GPU acceleration for faster inference

## Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ VRAM (recommended for quantized model)

## Installation

1. **Clone/Download the project**
   ```bash
   cd project3
   ```

2. **Install dependencies**
   ```bash
   pip install torch transformers peft fastapi uvicorn pydantic bitsandbytes
   ```

3. **Download the base model**
   - Download Phi-3 Mini from Hugging Face
   - Place it at: `C:\models\phi3_mini`
   - Ensure `tokenizer.json`, `config.json`, and model weights are present

4. **Verify adapter weights**
   - The project includes pre-trained LoRA adapters in `phi3-nmap-results/`
   - Adapters have been trained on nmap command generation tasks

## Project Structure

```
.
├── app.py                          # FastAPI application
├── load_Generate.py                # Data loading and generation utilities
├── nmap_dataset_corrected.json     # Training dataset
├── nmap_train.jsonl                # Training split
├── nmap_test.jsonl                 # Test split
├── Untitled1.ipynb                 # Jupyter notebook for experimentation
├── phi3-nmap-results/              # Fine-tuned LoRA adapters
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── checkpoint-200/
│   ├── checkpoint-400/
│   └── checkpoint-507/
└── README.md                       # This file
```

## Usage

### Starting the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
GET /
```

Response:
```json
{
  "status": "ok"
}
```

#### Generate Nmap Command
```bash
POST /generate
Content-Type: application/json

{
  "prompt": "Scan the network 192.168.1.0/24 for open ports"
}
```

Response:
```json
{
  "nmap_command": "nmap -p- -sV 192.168.1.0/24"
}
```

### Example Usage with cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Find all web servers in the network"}'
```

### Example Usage with Python

```python
import requests

url = "http://localhost:8000/generate"
payload = {"prompt": "Scan for SSH services on 10.0.0.0/8"}

response = requests.post(url, json=payload)
print(response.json())
```

## Model Configuration

### Base Model
- **Name**: Phi-3 Mini
- **Type**: Language Model for Causal Language Modeling
- **Size**: ~3.8B parameters (4-bit quantized)

### Quantization Settings
- **Load in 4-bit**: Yes
- **Quantization Type**: NF4 (Normal Float 4)
- **Compute Dtype**: float16
- **Double Quantization**: Enabled

### LoRA Adapter
- **Type**: Low-Rank Adaptation
- **Path**: `./phi3-nmap-results`
- **Training Data**: Nmap command generation tasks
- **Checkpoints Available**: 200, 400, 507 steps

## Performance Notes

- **Inference Speed**: ~1-2 seconds per command (GPU-dependent)
- **Memory Usage**: ~4-6GB VRAM with 4-bit quantization
- **Max Output Tokens**: 64 tokens per generation
- **Sampling**: Deterministic (do_sample=False) for reproducible results

## Configuration

Edit the following in `app.py` to customize:

```python
BASE_MODEL_PATH = r"C:\models\phi3_mini"  # Path to base model
ADAPTER_PATH = "./phi3-nmap-results"      # Path to LoRA adapters
```

## Training Data

The model was fine-tuned on:
- **nmap_dataset_corrected.json**: Full curated dataset
- **nmap_train.jsonl**: Training split with diverse nmap scenarios
- **nmap_test.jsonl**: Test split for evaluation

## Files Reference

- **app.py**: Main FastAPI application with model loading and inference
- **load_Generate.py**: Utilities for data loading and command generation
- **Untitled1.ipynb**: Jupyter notebook for interactive experimentation and testing

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_new_tokens` in the `generate_nmap()` function
- Use a smaller model or increase VRAM

### Model Not Found
- Verify `BASE_MODEL_PATH` points to the correct Phi-3 Mini directory
- Ensure all model files are present: `config.json`, `tokenizer.json`, weights

### Adapter Loading Error
- Verify `ADAPTER_PATH` is correct
- Ensure `adapter_config.json` and `adapter_model.safetensors` exist

## License

This project uses:
- **Phi-3**: Microsoft Research License
- **Transformers**: Apache 2.0
- **PEFT**: Apache 2.0

## Contributing

For improvements or issues:
1. Test changes on `nmap_test.jsonl`
2. Verify command quality with actual nmap
3. Update documentation accordingly

## References

- [Phi-3 Model Documentation](https://huggingface.co/microsoft/phi-3-mini)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [nmap Documentation](https://nmap.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
