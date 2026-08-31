"""
Scheduler for the nightly training pipeline.

Runs each task in config.yaml's `training.daily_schedule` at its
configured IST time using APScheduler's cron trigger.
"""

from typing import Any, Dict, Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.training.pipeline import TrainingPipeline


class TrainingScheduler:
    """Wires config.yaml's `training.daily_schedule` entries to TrainingPipeline tasks."""

    # classical_training / quantum_optimization / ensemble_optimization all map to
    # one combined HybridQMLModel.fit() call inside classical_and_quantum_training().
    TASK_MAP: Dict[str, str] = {
        "data_ingestion": "data_ingestion",
        "feature_engineering": "feature_engineering",
        "classical_training": "classical_and_quantum_training",
        "quantum_optimization": "classical_and_quantum_training",
        "ensemble_optimization": "classical_and_quantum_training",
        "backtest_validation": "backtest_validation",
        "model_deployment": "model_deployment",
    }

    # Tasks that share a method with an earlier task in the schedule should not
    # re-run that method; only the first occurrence actually fires it.
    _SKIP_DUPLICATE_METHOD = {"quantum_optimization", "ensemble_optimization"}

    def __init__(
        self,
        config: Dict[str, Any],
        pipeline: Optional[TrainingPipeline] = None,
        timezone: str = "Asia/Kolkata",
    ):
        self.config = config
        self.pipeline = pipeline or TrainingPipeline(config)
        self.scheduler = BlockingScheduler(timezone=timezone)

    def _run_task(self, task_name: str) -> None:
        method_name = self.TASK_MAP.get(task_name)
        if method_name is None:
            print(f"[scheduler] Unknown training task: {task_name}")
            return

        if task_name in self._SKIP_DUPLICATE_METHOD:
            print(f"[scheduler] Skipping {task_name} (already run as part of classical_training)")
            return

        method = getattr(self.pipeline, method_name)
        print(f"[scheduler] Running task: {task_name}")
        method()

    def start(self) -> None:
        for entry in self.config.get("training", {}).get("daily_schedule", []):
            task_name = entry["task"]
            hour, minute = (int(x) for x in entry["time"].split(":"))
            self.scheduler.add_job(
                self._run_task,
                trigger=CronTrigger(hour=hour, minute=minute),
                args=[task_name],
                id=task_name,
                name=task_name,
            )

        print("Training scheduler started. Jobs:", [job.id for job in self.scheduler.get_jobs()])
        self.scheduler.start()
