import numpy as np
from carbon_forecast.evaluation.metrics import EvaluationReport
import pandas as pd

def test_metrics_basic():
    """Check MAE and RMSE on known values"""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    # Errors 10, 10, 10 -> MAE = 10, RMSE = 10

    report = EvaluationReport(y_true=y_true, y_pred=y_pred)
    metrics = report.compute()

    assert metrics['mae'] == 10.0
    assert metrics['rmse'] == 10.0
    assert 'r2_score' in metrics

def test_picp():
    """
    Check Prediction Interval Coverage Probability

    PICP measures how often the true value 
    falls inside the prediction interval
    """

    y_true = np.array([100, 200, 300, 400])
    quantiles = pd.DataFrame({
        'q10': [90, 180, 290, 410],
        'q90': [110, 220, 310, 420]
    })

    # True values inside intervals:
    #
    # 100 -> inside
    # 200 -> inside
    # 300 -> inside
    # 400 -> outside (below q10=410)

    report = EvaluationReport(
        y_true=y_true,
        y_pred=y_true,
        quantile_preds=quantiles,
    )

    metrics = report.compute()

    assert metrics["picp"] == 0.75

def test_pinaw():
    """
    Check Prediction Interval Normalised Average Width.

    PINAW measures average interval width
    normalised by target range.

    Lower = sharper intervals.
    """

    y_true = np.array([100, 200, 300, 400])

    quantiles = pd.DataFrame({

        "q10": [90, 180, 290, 390],

        "q90": [110, 220, 310, 410],
    })

    # Interval widths:
    #
    # 20, 40, 20, 20
    #
    # Mean width = 25
    #
    # Target range:
    # 400 - 100 = 300
    #
    # PINAW:
    # 25 / 300 = 0.0833

    report = EvaluationReport(
        y_true=y_true,
        y_pred=y_true,
        quantile_preds=quantiles,
    )

    metrics = report.compute()

    assert round(metrics["pinaw"], 4) == 0.0833


def test_pinball_loss():
    """
    Check pinball loss computation.

    Pinball loss penalises quantile errors
    asymmetrically.

    Lower quantiles penalise overprediction more.
    Higher quantiles penalise underprediction more.
    """

    y_true = np.array([100, 200, 300])

    quantiles = pd.DataFrame({

        "q10": [90, 190, 290],

        "q90": [110, 210, 310],
    })

    report = EvaluationReport(
        y_true=y_true,
        y_pred=y_true,
        quantile_preds=quantiles,
    )

    metrics = report.compute()

    # Basic existence checks
    assert "pinball_q10" in metrics

    assert "pinball_q90" in metrics

    # Losses should be positive
    assert metrics["pinball_q10"] >= 0

    assert metrics["pinball_q90"] >= 0