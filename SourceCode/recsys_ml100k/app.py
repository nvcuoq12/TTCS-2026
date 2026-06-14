"""
app.py — FastAPI Backend
Chạy: uvicorn app:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os, time
from models import RecommendationSystem

app = FastAPI(title="CineRec - ML100K", version="1.0")

print("\n" + "="*55)
print("  CINEREC — MovieLens ml-100k  |  Đang khởi động...")
print("="*55)
t0 = time.time()
rs = RecommendationSystem("data/ratings.csv", "data/movies.csv")
print(f"  Khởi động xong trong {time.time()-t0:.1f}s")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

@app.get("/api/stats")
async def get_stats():
    return rs.stats()

@app.get("/api/users")
async def get_users():
    return {"user_ids": rs.all_users}

@app.get("/api/recommend/{user_id}")
async def recommend(user_id: int, method: str = "mf", n: int = 10):
    if user_id not in rs.uid2i:
        raise HTTPException(404, f"User {user_id} không tồn tại (range: 1–943)")
    if method not in ("ubcf", "ibcf", "mf"):
        raise HTTPException(400, "method phải là: ubcf | ibcf | mf")
    recs    = rs.recommend(user_id, method=method, n=n)
    history = rs.history(user_id)
    return {"user_id": user_id, "method": method,
            "recommendations": recs, "history": history}

@app.get("/api/health")
async def health():
    return {"status": "ok", "dataset": "ml-100k",
            "users": rs.stats()["n_users"], "movies": rs.stats()["n_movies"]}
