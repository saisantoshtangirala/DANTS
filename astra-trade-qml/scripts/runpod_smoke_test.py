"""
GPU smoke test for the Astra-Trade QML training path.

Runs a deliberately small end-to-end pass through the real training
pipeline (NSE ingestion -> feature engineering -> HybridQMLModel.fit()
-> backtest) on real torch/xgboost/qiskit, to verify the code actually
works — not to produce a usable trading model. Meant to be run once on a
GPU box (e.g. a RunPod pod) via `python3 scripts/runpod_smoke_test.py`.

Writes results to runpod_results/smoke_test_<timestamp>.json and .log so
they can be committed/pushed and inspected after the run.
"""

import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SYMBOLS = ["RELIANCE", "TCS"]
LOOKBACK_DAYS = 180
SYNTHETIC_ROWS = 220

log_lines = []


def log(msg: str) -> None:
    line = f"[{datetime.utcnow().isoformat()}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


def synthetic_ohlcv(symbol: str, n: int = SYNTHETIC_ROWS):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    volume = rng.integers(1_000, 100_000, n).astype(float)

    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def main() -> dict:
    results = {"started_at": datetime.utcnow().isoformat(), "symbols": SYMBOLS}

    try:
        import pandas as pd
        import torch

        from src.data.feature_engineering import FeatureConfig, FeatureEngineer
        from src.data.nse_ingestion import NSEDataIngestion
        from src.models.quantum.hybrid_model import HybridQMLModel
        from src.training.pipeline import build_hybrid_model_config
        from src.utils.config import load_config
        from src.utils.metrics import generate_performance_report

        results["torch_version"] = torch.__version__
        results["torch_cuda_available"] = torch.cuda.is_available()
        log(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")

        config = load_config()

        # 1. Try real NSE data first, fall back to synthetic if blocked/unavailable.
        ingestion = NSEDataIngestion(data_dir=str(REPO_ROOT / "data" / "nse"))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)

        raw = {}
        for symbol in SYMBOLS:
            try:
                df = ingestion.download_historical_range(symbol, start_date, end_date)
                if not df.empty and len(df) > 70:
                    raw[symbol] = df
                    log(f"NSE download OK for {symbol}: {len(df)} rows")
                else:
                    log(f"NSE download insufficient for {symbol}: {len(df)} rows")
            except Exception as e:
                log(f"NSE download failed for {symbol}: {e}")

        if raw:
            results["data_source"] = "nse_live"
        else:
            results["data_source"] = "synthetic_fallback"
            log("No usable NSE data; falling back to synthetic OHLCV")
            raw = {symbol: synthetic_ohlcv(symbol) for symbol in SYMBOLS}

        # 2. Feature engineering + labels
        fe = FeatureEngineer(FeatureConfig(lookback_periods=60))
        targets = config["signals"]["targets"]["intraday"]

        featured = {}
        for symbol, df in raw.items():
            feat = fe.generate_all_features(df)
            feat = fe.generate_labels(
                feat,
                profit_threshold=targets["profit_target_pct"],
                loss_threshold=-targets["stop_loss_pct"],
            )
            if not feat.empty:
                featured[symbol] = feat
                log(f"Featured {symbol}: {len(feat)} rows, {len(fe.get_feature_columns(feat))} features")

        if not featured:
            raise RuntimeError("No featured data produced for any symbol")

        pooled = pd.concat(list(featured.values()), ignore_index=True)
        feature_cols = fe.get_feature_columns(pooled)
        X = pooled[feature_cols].to_numpy()
        y = pooled["label"].to_numpy()

        split = int(len(X) * 0.8)
        X_train, y_train, X_val, y_val = X[:split], y[:split], X[split:], y[split:]
        log(f"Training matrix: train={X_train.shape} val={X_val.shape}")

        # 3. Build the hybrid model, overridden for smoke-test speed.
        # fallback_to_classical=False forces an actual attempt at the quantum
        # path (qkernel/VQC) instead of silently skipping it, since exercising
        # real qiskit code is the point of this run. Note the GPU only speeds
        # up the LSTM: quantum_kernel/vqc run AerSimulator's statevector method
        # on CPU regardless. FidelityQuantumKernel is O(n^2) circuit
        # evaluations in the training set size, so n_qubits/pca_components are
        # kept small (4, measured locally at ~100s for 150 samples) to keep
        # this bounded.
        hybrid_cfg = build_hybrid_model_config(config)
        hybrid_cfg["lstm"] = {**hybrid_cfg["lstm"], "epochs": 5, "early_stopping_patience": 3}
        hybrid_cfg["xgboost"] = {**hybrid_cfg["xgboost"], "n_estimators": 50}
        for key in ("quantum_kernel", "vqc"):
            hybrid_cfg[key] = {
                **hybrid_cfg[key],
                "n_qubits": 4,
                "pca_components": 4,
                "shots": 256,
                "fallback_to_classical": False,
            }
        hybrid_cfg["vqc"]["max_iter"] = 20

        model = HybridQMLModel(config=hybrid_cfg)
        sequence_length = min(60, max(1, len(X_train) - 1))

        t0 = time.time()
        fit_metrics = model.fit(
            X_train, y_train, X_val, y_val,
            feature_names=feature_cols,
            sequence_length=sequence_length,
        )
        train_seconds = time.time() - t0
        log(f"Training completed in {train_seconds:.1f}s")

        results["fit_metrics"] = fit_metrics
        results["quantum_metrics"] = model.get_quantum_metrics()
        results["train_seconds"] = train_seconds
        results["model_version"] = model.model_version

        # 4. Quick backtest on the held-out validation slice.
        oos_df = pooled.iloc[split:].reset_index(drop=True)
        predicted_labels = model.predict(X_val)
        confidence = model.get_signal_confidence(X_val)

        trade_returns = oos_df["future_return"].fillna(0.0) * predicted_labels
        trades_df = pd.DataFrame(
            {
                "pnl": trade_returns * 1_000_000,
                "pnl_pct": trade_returns,
                "confidence": confidence,
            }
        )
        equity_curve = (1 + trade_returns).cumprod() * 1_000_000
        results["backtest"] = generate_performance_report(trades_df, equity_curve)
        log(f"Backtest summary: {results['backtest']}")

        results["status"] = "SUCCESS"

    except Exception as e:
        results["status"] = "FAILED"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        log(f"FAILED: {e}\n{results['traceback']}")

    results["finished_at"] = datetime.utcnow().isoformat()
    return results


if __name__ == "__main__":
    outcome = main()

    out_dir = REPO_ROOT / "runpod_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    with open(out_dir / f"smoke_test_{stamp}.json", "w") as f:
        json.dump(outcome, f, indent=2, default=str)
    with open(out_dir / f"smoke_test_{stamp}.log", "w") as f:
        f.write("\n".join(log_lines))

    print(f"RESULT_STAMP={stamp}")
    sys.exit(0 if outcome.get("status") == "SUCCESS" else 1)
