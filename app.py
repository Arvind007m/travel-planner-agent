"""
FastAPI Backend Server for Trip Planner Agent Web UI
Serves the ChatGPT-like UI and runs the compiled LangGraph workflow.
"""

import os
import json
import time
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment
load_dotenv()

# Import the LangGraph workflow from main
import main as agent_module

app = FastAPI(title="Trip Planner Agent API", version="2.0")

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global compiled graph
agent_graph = agent_module.build_trip_planner_graph()

# In-memory session trace collector
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class ToolTraceItem(BaseModel):
    name: str
    args: Dict[str, Any]
    result: Any
    timestamp: str

@app.get("/api/health")
async def health_check():
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_amadeus = bool(os.getenv("AMADEUS_CLIENT_ID") and os.getenv("AMADEUS_CLIENT_SECRET"))
    return {
        "status": "healthy",
        "llm_model": "openai/gpt-oss-120b",
        "groq_connected": has_groq,
        "amadeus_connected": has_amadeus,
        "live_apis": ["Open-Meteo", "OpenStreetMap Nominatim", "Frankfurter", "Wikipedia"]
    }

@app.post("/api/chat")
async def handle_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    start_time = time.time()
    
    # Custom logger to capture tool execution traces for the UI
    captured_traces = []
    
    # Monkey patch log_tool_call temporarily for this request to capture traces
    original_log_tool_call = agent_module.log_tool_call
    def intercepting_log_tool_call(name: str, args: dict, result: Any):
        original_log_tool_call(name, args, result)
        captured_traces.append({
            "name": name,
            "args": args,
            "result": result,
            "timestamp": time.strftime("%H:%M:%S")
        })
        
    agent_module.log_tool_call = intercepting_log_tool_call
    
    try:
        initial_state = {
            "messages": [agent_module.HumanMessage(content=req.message)],
            "user_budget": None,
            "flight_cost": None,
            "hotel_cost": None,
            "total_cost": None,
            "tool_call_count": 0,
            "budget_retried": False,
            "retry_pending": False,
            "budget_status_note": None
        }
        
        result_state = agent_graph.invoke(initial_state)
        elapsed_sec = round(time.time() - start_time, 2)
        
        final_message = result_state["messages"][-1]
        reply_content = str(final_message.content)
        
        budget_info = {
            "user_budget": result_state.get("user_budget"),
            "flight_cost": result_state.get("flight_cost"),
            "hotel_cost": result_state.get("hotel_cost"),
            "total_cost": result_state.get("total_cost"),
            "budget_retried": result_state.get("budget_retried", False),
            "budget_status_note": result_state.get("budget_status_note")
        }
        
        return {
            "reply": reply_content,
            "traces": captured_traces,
            "budget": budget_info,
            "tool_call_count": result_state.get("tool_call_count", len(captured_traces)),
            "elapsed_seconds": elapsed_sec
        }
        
    except Exception as e:
        print(f"[API ERROR] {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        agent_module.log_tool_call = original_log_tool_call

# Serve Static UI files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Static index.html not yet created."})

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Trip Planner Agent Web UI on http://127.0.0.1:8000 ...\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
