# Afghanistan Crop AI — Full-stack ML Demonstration

React/Vite frontend + FastAPI backend + two real fitted scikit-learn pipelines.

## Features
- **Yield Prediction:** Random Forest regression for tons/hectare and total tons; RBF SVM for Low/Medium/High.
- **Fruit Recommendation:** compares predicted yield of all crops for the supplied scenario; **not** a validated suitability ranking.
- **Irrigation Analysis:** hypothetical model predictions for each irrigation type; **not** water-use estimates or irrigation advice.
- **Model Comparison:** previously observed classification test results from the project experiments.

The models were **retrained from the supplied 9,200-row synthetic dataset**, with the same split, feature lists and hyperparameters used in the class experiments. Reproduced SVM test accuracy: **89.02%**; regression MAE: **0.5277 t/ha**. Results are not real-world accuracy claims.

## Requirements
Python 3.11+; Node.js 20+.

## Run locally
In terminal 1:
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
In terminal 2:
```bash
cd frontend
npm install
# copy .env.example to .env for local use
npm run dev
```
Open `http://localhost:5173`. API docs: `http://localhost:8000/docs`.

## Publish with GitHub
1. Create a **public** empty repository on GitHub (e.g. `afghanistan-crop-ai`).
2. In this project directory run:
```bash
git init
git add .
git commit -m "Initial Afghanistan Crop AI full-stack app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/afghanistan-crop-ai.git
git push -u origin main
```
3. Model artifacts are compressed joblib files. Check GitHub's per-file size limits before pushing.

## Deploy backend on Render
1. In Render, create **New > Blueprint** and connect the GitHub repository; the `render.yaml` config deploys the backend Docker service.
2. Wait for `/api/health` to return `{ "status": "ok", ... }`.
3. Copy the public backend URL (e.g. `https://YOUR-BACKEND.onrender.com`).

## Deploy frontend on Vercel
1. Import the same GitHub repository in Vercel.
2. Set **Root Directory** to `frontend` and framework to **Vite**.
3. Set environment variable `VITE_API_URL` to the deployed backend URL **without trailing slash**.
4. Build command: `npm run build`; output directory: `dist`.
5. Deploy and visit the Vercel URL.

Backend CORS currently permits any origin for a public demo API; restrict to your deployed frontend origin for production. No user login or database is required for this read-only demo. Do not load untrusted `.joblib` files.

## API
- `GET /api/health`
- `GET /api/metadata`
- `POST /api/predict`
- `POST /api/recommend`
- `POST /api/irrigation`
- `GET /api/models`

Request JSON example:
```json
{"Year":2024,"Month":6,"Province":"Kandahar","Crop":"Pomegranates","Temp_C":26,"Rain_mm":250,"Fertilizer_kg_ha":100,"Irrigation_Type":"Drip","Land_Hectares":5}
```

**Limitation:** Predicted total production is regression yield × land area. The category model is independent and can disagree with numeric yield near thresholds. The scenario inputs are not a full agronomic suitability model. The historical accuracy chart includes a mix of standalone classifiers and a two-stage regression-to-category method.

## Selectable model registry (v2)

The frontend now provides two independent selectors on the Yield Prediction, Fruit Recommendation, and Irrigation Analysis screens:
- **Classification:** Gaussian Naive Bayes, Linear SVM, RBF SVM, KNN (K=31), Logistic Regression, tuned Decision Tree, Random Forest Classifier, and Random Forest Regression → Category.
- **Numeric yield:** Random Forest Regression, Linear Regression, Polynomial Regression (degree 2 with Ridge regularization).

The API returns the chosen model names and **held-out aggregate test metrics** with each prediction. Classification reports test accuracy and macro F1; regression reports MAE in tons/hectare and R². These are not the probability that a particular prediction is correct. The Model Comparison screen includes all models. All saved pipelines can be rebuilt from the included synthetic dataset using `cd backend && python train_all.py`. This regenerates `backend/models/registry.json` and the training-derived hybrid thresholds.

The original experiments also tested gradient-boosting regression with category conversion. This optional experimental approach is not included as a selectable deployed model, because its original experiment was cross-validation only and the current registry is designed around comparable held-out evaluations.
