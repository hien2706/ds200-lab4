import os
import json
import joblib
import logging
from typing import Dict
from sklearn.linear_model import SGDClassifier
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)

class LoanDefaultModelTrainer:
    """Train a loan default prediction model on batches of data"""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.model_count = 0
        self.model = self._load_or_create()
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)

    def _load_or_create(self):
        path = f"{self.output_dir}/models/model.joblib"
        if os.path.exists(path):
            logger.info(f"Loading existing model from {path}")
            return joblib.load(path)
        logger.info("Creating new SGDClassifier model")
        return SGDClassifier(loss="log_loss", penalty="l2", max_iter=1, warm_start=True)

    def train(self, pipeline, df: DataFrame, batch_id: int):
        logger.info(f"Training batch {batch_id} with {df.count()} rows")
        p = pipeline.fit(df)
        feat = p.transform(df)
        X = feat.select("features").rdd.map(lambda r: r[0].toArray()).collect()
        y = feat.select("label").rdd.map(lambda r: r[0]).collect()
        if X:
            self.model.partial_fit(X, y, classes=[0, 1])
        self.model_count += 1
        return p, self.model

    def evaluate(self, model, pipeline, df: DataFrame, batch_id: int) -> Dict:
        logger.info(f"Evaluating batch {batch_id}")
        p = pipeline.fit(df)
        feat = p.transform(df)
        preds = model.predict(feat.select("features").rdd.map(lambda r: r[0].toArray()).collect())
        trues = [r[0] for r in df.select("label").collect()]
        acc = sum(int(p==t) for p,t in zip(preds, trues)) / len(preds) if preds else 0
        metrics = {
            "batch_id": batch_id,
            "model_id": self.model_count,
            "accuracy": acc
        }
        return metrics

    def save_metrics(self, metrics: Dict, batch_id: int):
        fpath = f"{self.output_dir}/metrics/batch_{batch_id}_metrics.json"
        with open(fpath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved: {fpath}")

    def save_model(self, batch_id: int):
        path = f"{self.output_dir}/models/model_{self.model_count}_batch_{batch_id}.joblib"
        joblib.dump(self.model, path)
        logger.info(f"Model saved: {path}")