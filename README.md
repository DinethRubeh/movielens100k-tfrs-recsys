## MovieLens100k Ranked Recommendations using TFRS Ranking model (including side features)

1. Approach

2. Model Choice Rationale

3. How to run the code and reproduce results.
    - **Data pre-processing pipeline**: `src.preprocess.py`
        - commands:
            ```
            cd src
            python preprocess.py
            ```
    
    - **Model training pipeline**: `src.train.py`
        - commands:
            ```
            cd src
            python train.py
            ```

    - **Model inference (ranking logic) FastAPI service**: `src.app.py`
        - commands to run FastAPI server:
            ```
            cd src
            fastapi run app.py --host 0.0.0.0 --port 8000
            ```
        - Endpoint: `http://localhost:8000/recommendation/ranked_recommendation`
        - Request type: POST
        - Request Body:
            ```
            {
                "user_id": "52", 
                "candidate_movies": [
                    "Toy Story (1995)", 
                    "GoldenEye (1995)"
                    ]
            }
            ```
        
5. **Exploratory Data Analysis (EDA)** can be found here.
    - `notebooks.1_eda.ipynb`

4. Other Recommendation models trained can be found in the notebooks directory.
    - ***random recommender*** baseline
    - ***top-rated recommender*** baseline
    - content-based model using ***sentence transformer*** embeddings
    - ***collaborative filtering*** - item-based & user-based (librec library)
    - ***lightfm*** hybrid recommender
    - ***tfrs ranking*** model
    - ***tfrs ranking with side features*** model