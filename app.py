from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from model_pipeline import TEAMS, win_prob_latest

app = FastAPI()

class PredictRequest(BaseModel):
    home_team: str
    away_team: str

@app.get("/api/teams")
def get_teams():
    return {"teams": TEAMS}

@app.post("/api/predict")
def predict(req: PredictRequest):
    out = win_prob_latest(req.home_team, req.away_team)
    return {
        **out,
        "home_win_pct": round(out["home_win_prob"] * 100, 2),
        "away_win_pct": round(out["away_win_prob"] * 100, 2),
    }

@app.get("/")
def home():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")
