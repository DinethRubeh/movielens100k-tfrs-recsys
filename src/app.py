from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.inference import load_user_movie_data, load_model, recommend_ranked_candidates_for_user
from src.endpoints.models import RankedRecommendation

app = FastAPI(title="Ranked Movie Recommendations")

tfrs_model = load_model()
users_df, movies_df = load_user_movie_data()

@app.get("/recommendation")
def health_check():
    return JSONResponse(
        content={
            'code':200,
            'response':'health check: OK'
            })

@app.post("/recommendation/ranked_recommendation")
def get_ranked_recommendations(request: RankedRecommendation):
    user_id = request.user_id
    candidate_movies = request.candidate_movies

    ranked_candidates = recommend_ranked_candidates_for_user(tfrs_model, user_id, candidate_movies)
    return JSONResponse(
        content={
            'code':200,
            'response':ranked_candidates
            })