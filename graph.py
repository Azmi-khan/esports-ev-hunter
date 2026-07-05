from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import requests

load_dotenv()

class EVTrackerState(TypedDict):
    upcoming_matches: List[Dict[str, Any]]
    bookmaker_odds: List[Dict[str, Any]]
    ev_opportunities: List[Dict[str, Any]]
    email_html: str

 #agent 1
def stat_puller_node(state: EVTrackerState):
    print("\n--- AGENT 1: Fetching Match Statistics & Market Odds ---")

    mock_matches = [
        {"match_id": "CS2_101", "team_a": "FaZe", "team_b": "NAVI", "faze_win_prob": 0.60},
        {"match_id": "LOL_202", "team_a": "T1", "team_b": "Gen.G", "t1_win_prob": 0.45}
    ]

    mock_odds = [
        {"match_id": "CS2_101", "bookie": "Pinnacle", "team_a_odds": 2.10},  # Implied prob: 1/2.10 = 47.6%
        {"match_id": "LOL_202", "bookie": "Bet365", "team_a_odds": 1.85}  # Implied prob: 1/1.85 = 54.0%
    ]

    # We update the state dictionary with our raw data arrays
    return {"upcoming_matches": mock_matches, "bookmaker_odds": mock_odds}

#agent 2
def risk_modeler_node(state: EVTrackerState):
    print("\n--- AGENT 2: Executing Expected Value (+EV) Algorithms ---")
    matches = state.get("upcoming_matches", [])
    odds = state.get("bookmaker_odds", [])

    ev_bets = []

    # Iterating through matches and bookie listings to calculate structural anomalies
    for match in matches:
        for odd in odds:
            if match["match_id"] == odd["match_id"]:
                # Calculate what probability the bookie is predicting
                implied_prob = 1 / odd["team_a_odds"]

                # Dynamic matching based on team tags
                if "faze_win_prob" in match:
                    real_prob = match["faze_win_prob"]
                    team_name = match["team_a"]
                elif "t1_win_prob" in match:
                    real_prob = match["t1_win_prob"]
                    team_name = match["team_a"]
                else:
                    continue

                # Fundamental Quant Target: Real Probability > Implied Probability = Positive Value
                if real_prob > implied_prob:
                    edge_percentage = (real_prob - implied_prob) * 100
                    ev_bets.append({
                        "match_id": match["match_id"],
                        "team": team_name,
                        "odds": odd["team_a_odds"],
                        "edge_percent": round(edge_percentage, 2),
                        "recommendation": "PLAY"
                    })

    print(f"[Quant System Alert] Identified {len(ev_bets)} mispriced market opportunities.")
    return {"ev_opportunities": ev_bets}

# agent 3
def email_synthesizer_node(state: EVTrackerState):
    print("\n--- AGENT 3: Processing Generative HTML Delivery Format ---")
    ev_bets = state.get("ev_opportunities", [])

    if not ev_bets:
        return {"email_html": "<p>No statistical anomalies found in current market window.</p>"}

    # Initialize Gemini. It calls GOOGLE_API_KEY invisibly from the environment context
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    prompt = PromptTemplate.from_template("""
    You are an elite quantitative esports analyst working for a private fund. Take the following +EV betting opportunities 
    and format them into a clean, aggressive, and highly professional HTML email report.

    Data Source Payload: {ev_data}

    Output parameters:
    1. Output strictly valid, functional raw HTML.
    2. Emphasize a sleek dark-mode trading desk aesthetic.
    3. Construct a clear table showing Match ID, Team, Odds, and Edge.
    4. Visually distinguish the 'Edge %' metric using green styling cues to denote profit metrics.
    5. Do not wrap the response in markdown code blocks like ```html. Output the naked HTML starting directly with your tags.
    """)

    response = llm.invoke(prompt.format(ev_data=ev_bets))

    return {"email_html": response.content}

# Initializing workflow with our schema structure
workflow = StateGraph(EVTrackerState)

# Pinning our modular agent nodes to the workflow graph board
workflow.add_node("stat_puller", stat_puller_node)
workflow.add_node("risk_modeler", risk_modeler_node)
workflow.add_node("email_synthesizer", email_synthesizer_node)

# Wiring the execution paths sequentially
workflow.add_edge(START, "stat_puller")
workflow.add_edge("stat_puller", "risk_modeler")
workflow.add_edge("risk_modeler", "email_synthesizer")
workflow.add_edge("email_synthesizer", END)

# Compiling into a unified executable app instance
app = workflow.compile()

# This allows us to run the file directly to test it
# if __name__ == "__main__":
#     print("Initializing test run...")
#     # We pass an empty state dictionary because Agent 1 will populate it with the mock data
#     initial_state = {
#         "upcoming_matches": [],
#         "bookmaker_odds": [],
#         "ev_opportunities": [],
#         "email_html": ""
#     }
#
#     # Run the graph
#     final_output = app.invoke(initial_state)
#
#     print("\n=====================================================================")
#     print("FINAL GENERATED HTML OUTPUT FROM GEMINI:")
#     print("=====================================================================")
#     print(final_output["email_html"])