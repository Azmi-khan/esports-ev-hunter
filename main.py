from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from graph import app as langgraph_app

app = FastAPI(title = "Esports +EV Quant API")

class TriggerRequest(BaseModel):
    trigger_reason: str = "scheduled_daily_run"

@app.post("/generate-report")
async def generate_report(request: TriggerRequest):
    print(f"\n[SERVER] API hit! trigger reason: {request.trigger_reason}")
    print("[SERVER] spinning up LangGraph Engine...")


    inital_state = {
        "upcoming_matches": [],
        "bookmaker_odds": [],
        "ev_opportunities": [],
        "email_html" : ""
    }

    final_output = langgraph_app.invoke(inital_state)

    print("[SERVER] Graph execution complete. Returning payload.")

    return{
        "status": "success",
        "bets_found": len(final_output["ev_opportunities"]),
        "html_email": final_output["email_html"]
    }
# 4. Local Execution Block
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)