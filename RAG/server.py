from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.rag_agent import NmapRagAgent
import uvicorn

app = FastAPI(title="RAG API", description="RAG Agent API for Nmap command generation")

# Enable CORS for all origins and all IP addresses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = NmapRagAgent()


class CommandRequest(BaseModel):
    """Request model for command generation"""
    query: str
    target: str


class CommandResponse(BaseModel):
    """Response model for command generation"""
    status: str
    command: str
    intent: str
    target: str
    agent: str
    confidence: float


@app.post("/generate_command", response_model=CommandResponse)
async def generate_command(request: CommandRequest):
    """
    Generate Nmap command from query and target
    
    Receives: {"query": "...", "target": "..."}
    Returns: {"command": "nmap...", "intent": "...", "agent": "RAG", ...}
    """
    try:
        # Combine query and target into a single prompt
        full_query = f"{request.query} on {request.target}" if request.target else request.query
        
        # Generate command using process() method which returns a dict
        result = rag.process({
            "user_query": full_query,
            "extracted_ip": request.target
        })
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error_message", "Generation failed"))
        
        command = result.get("nmap_candidate", "")
        
        return CommandResponse(
            status="success",
            command=command,
            intent=request.query,
            target=request.target,
            agent="RAG",
            confidence=0.8
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "agent": "RAG"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all IP addresses
        port=8000
    )