from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from RAG.agent.rag_agent import RAGAgent
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

rag = RAGAgent()


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
        # Generate command
        result = rag.generate(request.query, request.target)
        
        return CommandResponse(
            status="success",
            command=result["command"],
            intent=result["intent"],
            target=request.target,
            agent="RAG",
            confidence=result.get("confidence", 0.8)
        )
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