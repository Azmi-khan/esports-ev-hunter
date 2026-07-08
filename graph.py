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
    print("\n--- AGENT 1: Fetching LIVE Market Odds ---")
    api_key = os.getenv("ODDS_API_KEY")

    if not api_key:
        print("[ERROR] ODDS_API_KEY missing from .env file!")
        return state

    # STEP 1: Reconnaissance - Ask the API what sports are actively running right now
    sports_url = "https://api.the-odds-api.com/v4/sports"
    sports_response = requests.get(sports_url, params={"apiKey": api_key})

    if sports_response.status_code != 200:
        print(f"[ERROR] Could not fetch sports list: {sports_response.text}")
        return {"upcoming_matches": [], "bookmaker_odds": []}

    all_sports = sports_response.json()

    # Filter the massive list down to just active Esports
    active_esports = [
        s for s in all_sports
        if "esports" in s.get("group", "").lower() or "esports" in s.get("key", "").lower()
    ]

    if not active_esports:
        print("[AGENT 1] No active esports tournaments found on the board today.")
        return {"upcoming_matches": [], "bookmaker_odds": []}

    # Grab the exact key for the first active tournament it finds
    target_sport_key = active_esports[0]["key"]
    print(f"[AGENT 1] Discovered active tournament: '{target_sport_key}'. Locking on...")

    # STEP 2: Execution - Fetch the odds specifically for that active tournament
    odds_url = f"https://api.the-odds-api.com/v4/sports/{target_sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",  # Widened the net to catch more bookies
        "markets": "h2h",
        "bookmakers": "pinnacle,bet365"
    }

    odds_response = requests.get(odds_url, params=params)

    if odds_response.status_code != 200:
        print(f"[ERROR] API Request Failed: {odds_response.text}")
        return {"upcoming_matches": [], "bookmaker_odds": []}

    raw_data = odds_response.json()
    print(f"[AGENT 1] Successfully pulled {len(raw_data)} live matches.")

    return {"upcoming_matches": raw_data, "bookmaker_odds": []}


#agent 2
def risk_modeler_node(state: EVTrackerState):
    print("\n--- AGENT 2: Executing Expected Value (+EV) Algorithms ---")
    matches = state.get("upcoming_matches", [])

    ev_bets = []

    for match in matches:
        team_a = match.get("home_team")
        team_b = match.get("away_team")

        pinnacle_odds = None
        bet365_odds = None

        for bookmaker in match.get("bookmakers", []):
            if bookmaker["key"] == "pinnacle":

                pinnacle_odds = bookmaker["markets"][0]["outcomes"][0]["price"]
            elif bookmaker["key"] == "bet365":
                bet365_odds = bookmaker["markets"][0]["outcomes"][0]["price"]


        if pinnacle_odds and bet365_odds:
            true_prob = 1 / pinnacle_odds
            implied_prob = 1 / bet365_odds


            if true_prob > implied_prob:
                edge_percentage = (true_prob - implied_prob) * 100


                if edge_percentage > 1.0:
                    ev_bets.append({
                        "match_id": f"{team_a} vs {team_b}",
                        "team": team_a,
                        "odds": bet365_odds,
                        "edge_percent": round(edge_percentage, 2),
                        "recommendation": "PLAY (Bet365 Misprice)"
                    })

    print(f"[Quant System Alert] Identified {len(ev_bets)} live mispriced opportunities.")
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