"""
Run the LLM failure analysis agent on saved model predictions.

This script is separate from run_training.py because:
    - Agent analysis involves API calls (cost + rate limits)
    - Doesn't need to re-run every training cycle
    - Can iterate on prompts/tools without retraining

Usage:
    python scripts/run_agent.py
"""

import os
import json
import logging
import pandas as pd
from carbon_forecast.agent.analyst_claude import FailureAnalyst
from carbon_forecast.agent.tools import AnalysisTools
from carbon_forecast.evaluation.failure import FailureDetector
from carbon_forecast.evaluation.stratified import StratifiedEvaluator
from carbon_forecast.evaluation.metrics import EvaluationReport
from carbon_forecast.models.lgbm import LGBMForecaster
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config_path = os.path.join(PROJECT_ROOT, "configs", "model", "lgbm_24h.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

HORIZON = config["horizon"]


def main():
    # -----------------------------------
    # LOAD SAVED PREDICTIONS
    # -----------------------------------

    preds_path = os.path.join(PROJECT_ROOT, "outputs", "predictions", "latest.parquet")
    eval_path = os.path.join(PROJECT_ROOT, "outputs", "predictions", "evaluation_df.parquet")

    if not os.path.exists(preds_path):
        raise FileNotFoundError(
            f"No predictions found at {preds_path}. Run run_training.py first."
        )

    preds_df = pd.read_parquet(preds_path)
    y_true = preds_df["y_true"]
    y_pred = preds_df["y_pred"]

    evaluation_df = pd.read_parquet(eval_path)

    logger.info(f"Loaded {len(y_true):,} predictions")

    # -----------------------------------
    # REBUILD EVALUATION (from saved data)
    # -----------------------------------

    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", f"lgbm_t{HORIZON}.pkl")
    model = LGBMForecaster.load(model_path)
    importance = model.feature_importance

    failure = FailureDetector(y_true=y_true, y_pred=y_pred, df=evaluation_df)
    failure_df = failure.report()

    evaluator = StratifiedEvaluator(
        df=evaluation_df,
        y_true=y_true,
        y_pred=y_pred,
    )
    stratified_results = evaluator.report()

    # -----------------------------------
    # RUN AGENT
    # -----------------------------------

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    analysis_tools = AnalysisTools(
        y_true=y_true,
        y_pred=y_pred,
        df=evaluation_df,
    )

    analyst = FailureAnalyst(
        tools=analysis_tools,
        api_key=api_key,
    )

    agent_report = analyst.analyse(
        failure_report=failure_df.head(10).to_dict(orient="records"),
        stratified_report=stratified_results.head(15).to_dict(orient="records"),
        feature_importances={k: int(v) for k, v in importance.items()},
        available_features=evaluation_df.columns.tolist(),
    )

    # -----------------------------------
    # SAVE REPORT
    # -----------------------------------

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(PROJECT_ROOT, "outputs", "reports")
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, f"agent_analysis_{version}.json")
    with open(report_path, "w") as f:
        json.dump(agent_report, f, indent=2, default=str)

    logger.info(f"Agent analysis saved to {report_path}")


if __name__ == "__main__":
    main()