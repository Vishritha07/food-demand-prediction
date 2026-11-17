import io
import os
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in backend/.env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Platewise ML Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MANDATORY_COLUMNS = [
    "Date",
    "Day",
    "Attendance",
    "Event",
    "Food Item",
    "Food Prepared",
    "Food Wasted",
]


def extract_numeric(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    match = re.search(r"[-+]?\d*\.?\d+", str(val))
    return float(match.group()) if match else np.nan


def normalize_text(val):
    if pd.isna(val):
        return ""
    return " ".join(str(val).strip().split()).title()


class PredictRequest(BaseModel):
    target_date: Optional[str] = None
    day_of_week: Optional[str] = None
    attendance: int
    event: str = "Regular"
    food_item: str
    extra_factors: Optional[Dict[str, Any]] = None

@app.get("/models/{canteen_id}")
@app.get("/dataset/{canteen_id}")
def get_dataset_status(canteen_id: str):
    res = supabase.table("canteen_models").select("*").eq("canteen_id", canteen_id).execute()
    if not res.data:
        return {"has_model": False}
    return {"has_model": True, "model": res.data[0]}


@app.delete("/dataset/{canteen_id}")
def delete_dataset(canteen_id: str):
    model_filename = f"{canteen_id}.joblib"
    try:
        supabase.storage.from_("models").remove([model_filename])
    except Exception:
        pass

    supabase.table("canteen_models").delete().eq("canteen_id", canteen_id).execute()
    return {"message": "Existing dataset and trained model deleted successfully."}


@app.post("/train/{canteen_id}")
async def train_model(canteen_id: str, file: UploadFile = File(...)):
    existing = supabase.table("canteen_models").select("model_id").eq("canteen_id", canteen_id).execute()
    if existing.data:
        raise HTTPException(
            status_code=400,
            detail="An active historical dataset already exists. Delete the existing dataset before uploading a replacement.",
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]
        if "Food Waste" in df.columns and "Food Wasted" not in df.columns:
            df = df.rename(columns={"Food Waste": "Food Wasted"})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    missing_cols = [col for col in MANDATORY_COLUMNS if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing mandatory columns (case-sensitive): {', '.join(missing_cols)}",
        )

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=MANDATORY_COLUMNS)

    df["Day"] = df["Day"].apply(normalize_text)
    df["Event"] = df["Event"].apply(normalize_text)
    df["Food Item"] = df["Food Item"].apply(normalize_text)

    df["Attendance"] = df["Attendance"].apply(extract_numeric)
    df["Food Prepared"] = df["Food Prepared"].apply(extract_numeric)
    df["Food Wasted"] = df["Food Wasted"].apply(extract_numeric)

    df = df.dropna(subset=["Attendance", "Food Prepared", "Food Wasted"])
    df = df[(df["Attendance"] > 0) & (df["Food Prepared"] >= 0) & (df["Food Wasted"] >= 0)]

    if len(df) < 5:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain at least 5 valid, non-duplicate rows after cleaning.",
        )

    df["Actual_Demand"] = (df["Food Prepared"] - df["Food Wasted"]).clip(lower=0)

    base_features = ["Day", "Attendance", "Event", "Food Item"]
    extra_cols = [c for c in df.columns if c not in MANDATORY_COLUMNS and c != "Actual_Demand"]

    valid_extra_cols = []
    for col in extra_cols:
        if df[col].isna().mean() < 0.3:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown").apply(normalize_text)
            valid_extra_cols.append(col)

    candidate_features = base_features + valid_extra_cols

    def build_pipeline(feature_list):
        cat_cols = [c for c in feature_list if not pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in feature_list if pd.api.types.is_numeric_dtype(df[c])]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

    selected_features = list(base_features)
    if valid_extra_cols:
        temp_pipe = build_pipeline(candidate_features)
        temp_pipe.fit(df[candidate_features], df["Actual_Demand"])
        importances = temp_pipe.named_steps["regressor"].feature_importances_
        encoded_names = temp_pipe.named_steps["preprocessor"].get_feature_names_out()

        for col in valid_extra_cols:
            col_imp = sum(
                imp for name, imp in zip(encoded_names, importances) if col in name
            )
            if col_imp >= 0.01:
                selected_features.append(col)

    demand_pipeline = build_pipeline(selected_features)
    demand_pipeline.fit(df[selected_features], df["Actual_Demand"])

    wastage_pipeline = build_pipeline(selected_features)
    wastage_pipeline.fit(df[selected_features], df["Food Wasted"])

    unique_food_items = sorted(df["Food Item"].unique().tolist())

    artifact = {
        "demand_model": demand_pipeline,
        "wastage_model": wastage_pipeline,
        "features": selected_features,
        "food_items": unique_food_items,
    }

    local_path = f"{canteen_id}.joblib"
    model_storage_path = f"{canteen_id}.joblib"
    joblib.dump(artifact, local_path)

    try:
        with open(local_path, "rb") as f:
            supabase.storage.from_("models").upload(
                path=model_storage_path,
                file=f,
                file_options={"content-type": "application/octet-stream", "upsert": "true"},
            )
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

    supabase.table("canteen_models").insert(
        {
            "canteen_id": canteen_id,
            "model_path": model_storage_path,
            "food_items": unique_food_items,
            "row_count": len(df),
            "dataset_filename": file.filename,
        }
    ).execute()

    return {
        "message": "Random Forest model trained successfully.",
        "rows_processed": len(df),
        "food_items": unique_food_items,
        "selected_features": selected_features,
    }


@app.post("/predict/{canteen_id}")
def predict_demand(canteen_id: str, req: PredictRequest):
    res = supabase.table("canteen_models").select("model_path").eq("canteen_id", canteen_id).execute()
    if not res.data:
        raise HTTPException(
            status_code=404,
            detail="No trained model found for this canteen. Please upload historical data first.",
        )

    model_path = res.data[0]["model_path"]

    try:
        file_bytes = supabase.storage.from_("models").download(model_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Model file missing in storage. Please re-upload dataset.")

    artifact = joblib.load(io.BytesIO(file_bytes))
    demand_model = artifact["demand_model"]
    wastage_model = artifact["wastage_model"]
    features = artifact["features"]

    day_val = req.day_of_week
    if not day_val and req.target_date:
        day_val = datetime.strptime(req.target_date, "%Y-%m-%d").strftime("%A")
    if not day_val:
        day_val = "Monday"

    row_data = {
        "Day": normalize_text(day_val),
        "Attendance": req.attendance,
        "Event": normalize_text(req.event),
        "Food Item": normalize_text(req.food_item),
    }

    if req.extra_factors:
        for k, v in req.extra_factors.items():
            row_data[k] = normalize_text(v) if isinstance(v, str) else v

    for feat in features:
        if feat not in row_data:
            row_data[feat] = "Unknown"

    input_df = pd.DataFrame([row_data])[features]

    pred_demand = int(round(max(0.0, float(demand_model.predict(input_df)[0]))))

    supabase.table("predictions_history").insert(
        {
            "canteen_id": canteen_id,
            "target_date": req.target_date,
            "day_of_week": row_data["Day"],
            "attendance": req.attendance,
            "event": row_data["Event"],
            "food_item": row_data["Food Item"],
            "predicted_demand": pred_demand,
            "predicted_wastage": 0,
        }
    ).execute()

    return {
        "food_item": row_data["Food Item"],
        "food_to_prepare": pred_demand,
        "unit": "plates",
    }
