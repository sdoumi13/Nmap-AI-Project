
"""
Steps 4-6: MCP Protocol, Server, Client
"""

import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# Step 4: Protocol (Message Schemas)
@dataclass
class ValidationRequest:
    command: str
    intent: str
    context: Dict[str, Any]

@dataclass
class ValidationResponse:
    valid: bool
    score: float
    status: str
    errors: List[str]
    warnings: List[str]
    method_used: str
