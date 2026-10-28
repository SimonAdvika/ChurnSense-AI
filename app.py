"""ChurnSense AI - FastAPI backend serving the FT-Transformer churn model.
Run:  pip install -r requirements.txt  &&  uvicorn app:app --reload
"""
import os, pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
MODELS = BASE / "models"
meta = pickle.load(open(MODELS / "ft_transformer_metadata.pkl", "rb"))
pre = joblib.load(MODELS / "telecom_preprocessor.pkl")  # needs scikit-learn==1.6.1

CAT, VOCAB = meta["categorical_features"], meta["vocab"]
STD, MM = meta["standard_features"], meta["minmax_features"]
THRESHOLD = meta["threshold"]
cfg = meta["model_config"]
D = cfg["d_token"]


class FTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_weight = nn.Parameter(torch.zeros(meta["n_numeric"], D))
        self.num_bias = nn.Parameter(torch.zeros(meta["n_numeric"], D))
        self.cat_embeddings = nn.ModuleList(nn.Embedding(c, D) for c in meta["cardinalities"])
        self.cat_bias = nn.Parameter(torch.zeros(len(CAT), D))
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        layer = nn.TransformerEncoderLayer(
            D, cfg["n_heads"], 4 * D, cfg["dropout"], batch_first=True,
            # Architecture details not stored in the files; override via env if outputs look wrong.
            activation=os.getenv("FT_ACTIVATION", "relu"),
            norm_first=os.getenv("FT_NORM_FIRST", "0") == "1",
        )
        self.encoder = nn.TransformerEncoder(layer, cfg["n_layers"], enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))  # activation has no weights

    def forward(self, x_num, x_cat):
        n = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        c = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.cat_embeddings)], 1) + self.cat_bias
        t = torch.cat([self.cls.expand(len(x_num), -1, -1), n, c], 1)
        return self.head(self.encoder(t)[:, 0]).squeeze(-1)


model = FTTransformer()
state = torch.load(MODELS / "best_ft_transformer.pt", map_location="cpu")
model.load_state_dict(state.get("state_dict", state))
model.eval()

std_tf = pre.named_transformers_["standard"]
mm_tf = pre.named_transformers_["minmax"]


def tenure_group(m: float) -> str:
    return ("0-6 months" if m <= 6 else "7-12 months" if m <= 12 else "13-24 months" if m <= 24
            else "25-48 months" if m <= 48 else "49+ months")


def build_row(d: dict) -> dict:
    """Fill derived columns the form doesn't ask for."""
    t = max(float(d["Tenure in Months"]), 1)
    d["Total Charges"] = float(d["Monthly Charge"]) * t
    d["Total Revenue"] = (d["Total Charges"] - float(d["Total Refunds"])
                          + float(d["Total Extra Data Charges"]) + float(d["Total Long Distance Charges"]))
    d["AverageMonthlySpend"] = d["Total Revenue"] / t
    d["TenureGroup"] = tenure_group(t)
    return d


app = FastAPI(title="ChurnSense AI")


@app.post("/predict")
def predict(d: dict):
    try:
        df = pd.DataFrame([build_row(dict(d))])
        num = np.hstack([std_tf.transform(df[STD]), mm_tf.transform(df[MM])]).astype("float32")
        cat = [[VOCAB[c].get(str(df[c].iloc[0]), 0) for c in CAT]]
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(num), torch.tensor(cat))).item()
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return {"probability": p, "threshold": THRESHOLD, "churn": p >= THRESHOLD}


@app.get("/health")
def health():
    return {"status": "ok", "val_auc": meta["best_validation_auc"], "threshold": THRESHOLD}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
