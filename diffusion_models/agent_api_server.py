"""
FastAPI Server (Single Endpoint)
Input: natural language query
Output: generated Nmap command
Uses discrete_diffusion_nmap model directly
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
from discrete_diffusion_nmap import NmapDiscreteDiffusionLM, DiscreteDiffusionSampler

app = FastAPI(title="Nmap Diffusion API", version="1.0")

# Initialize model and sampler once
checkpoint_path = 'nmap_diffusion_checkpoint'
try:
    model = NmapDiscreteDiffusionLM(model_name=checkpoint_path, use_adapter=False)
except Exception:
    # Fallback to base model if checkpoint missing
    model = NmapDiscreteDiffusionLM(model_name='t5-small', use_adapter=False)
sampler = DiscreteDiffusionSampler(model, max_steps=10)

class GenerateRequest(BaseModel):
    query: str

class GenerateResponse(BaseModel):
    command: str

@app.post("/generate", response_model=GenerateResponse)
def generate_command(request: GenerateRequest):
    """Generate an Nmap command from a natural language query."""
    try:
        result = sampler.sample(request.query, verbose=False)
        return GenerateResponse(command=result['final_command'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
