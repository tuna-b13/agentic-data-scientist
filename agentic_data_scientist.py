"""
Agentic Data Scientist — Orchestrator
Coordinates the full pipeline: load -> profile -> plan -> preprocess ->
train -> evaluate -> reflect -> (replan loop) -> report.

All decisions are driven by data signals from the profiler, not hardcoded.
"""

import os
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from agents.planner import create_plan, explain_plan
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory import JSONMemory
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint
from tools.modelling import build_preprocessor, select_models, train_models
from tools.evaluation import evaluate_best, write_markdown_report, save_json


@dataclass
class RunContext:
    run_id: str
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class AgenticDataScientist:
    """
    Offline Agentic Data Scientist.

    Pipeline:
      1. Load dataset
      2. Infer target (if auto)
      3. Profile dataset (rich signals)
      4. Check memory for past runs
      5. Create adaptive plan
      6. Build preprocessor, select & train models
      7. Evaluate best model
      8. Reflect on results
      9. If reflection recommends replan -> modify plan & repeat
     10. Save all artefacts and update memory
    """

    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
        self.verbose = verbose
        self.memory = JSONMemory(memory_path)
        self.ctx: Optional[RunContext] = None
        self.state: Dict[str, Any] = {}
        self.logger = logging.getLogger("AgenticDS")

    def _setup_logging(self, output_dir: str) -> None:
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)
        if self.verbose:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

    def log(self, msg: str, level: str = "info") -> None:
        getattr(self.logger, level, self.logger.info)(msg)

    def load_data(self, path: str) -> pd.DataFrame:
        self.log(f"Loading dataset: {path}")
        df = pd.read_csv(path, sep=None, engine="python")
        self.log(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")
        return df

    def run(
        self,
        data_path: str,
        target: str,
        output_root: str = "outputs",
        seed: int = 42,
        test_size: float = 0.2,
        max_replans: int = 2,
    ) -> str:
        # Setup
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        self._setup_logging(output_dir)
        self.log(f"Run {run_id} started")

        self.ctx = RunContext(
            run_id=run_id,
            started_at=now_iso(),
            data_path=data_path,
            target=target,
            output_dir=output_dir,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )
        self.state = {"replan_count": 0}

        # Step 1: Load data
        df = self.load_data(data_path)

        # Step 2: Infer target if auto
        if target.strip().lower() == "auto":
            inferred = infer_target_column(df)
            if not inferred:
                raise ValueError("Could not infer target column. Please provide --target <name>.")
            self.ctx.target = inferred
            self.log(f"Auto-detected target: {inferred}")

        # Step 3: Profile dataset
        self.log("Profiling dataset...")
        profile = profile_dataset(df, self.ctx.target)
        fp = dataset_fingerprint(df, self.ctx.target)
        self.log(f"Profile: {profile['shape']['rows']} rows, "
                 f"{len(profile['feature_types']['numeric'])} numeric, "
                 f"{len(profile['feature_types']['categorical'])} categorical, "
                 f"imbalance={profile.get('imbalance_ratio')}, "
                 f"missing={profile.get('total_missing_pct', 0):.1f}%")

        # Save EDA summary early
        save_json(os.path.join(output_dir, "eda_summary.json"), profile)

        # Step 4: Check memory
        prev = self.memory.get_dataset_record(fp)
        if prev:
            self.log(f"Memory hit: previous best={prev.get('best_model')} "
                     f"(F1={prev.get('best_metrics', {}).get('f1_macro', '?')})")
        else:
            self.log("No memory for this dataset fingerprint.")

        # Step 5: Create plan
        plan_dict = create_plan(profile, memory_hint=prev)
        steps = plan_dict["steps"]
        self.log(f"Plan ({len(steps)} steps):")
        self.log(explain_plan(plan_dict))

        # Save plan
        save_json(os.path.join(output_dir, "plan.json"), {
            "steps": steps,
            "rationale": plan_dict["rationale"],
            "hints": plan_dict["hints"],
            "explanation": explain_plan(plan_dict),
        })

        # Step 6-9: Execute loop
        iteration = 0
        while True:
            iteration += 1
            self.log(f"=== Iteration {iteration} ===")

            # Build preprocessor
            preprocessor = build_preprocessor(profile)

            # Select models
            candidates = select_models(profile, seed=self.ctx.seed)
            self.log(f"Models: {[n for n, _ in candidates]}")

            # Train
            results = train_models(
                df=df,
                target=self.ctx.target,
                preprocessor=preprocessor,
                candidates=candidates,
                seed=self.ctx.seed,
                test_size=self.ctx.test_size,
                output_dir=output_dir,
                verbose=self.verbose,
                profile=profile,
            )

            # Evaluate
            eval_payload = evaluate_best(results, output_dir=output_dir)
            best = eval_payload["best_metrics"]
            is_clf = eval_payload.get("is_classification", True)
            if is_clf:
                self.log(f"Best: {best['model']} | "
                         f"BalAcc={best['balanced_accuracy']:.3f} | "
                         f"F1={best['f1_macro']:.3f}")
            else:
                self.log(f"Best: {best['model']} | "
                         f"R2={best['r2']:.3f} | "
                         f"RMSE={best['rmse']:.3f}")

            # Reflect
            reflection = reflect(
                dataset_profile=profile,
                evaluation=best,
                all_metrics=eval_payload["all_metrics"],
            )
            self.log(f"Reflection: status={reflection['status']}, "
                     f"issues={reflection['n_issues']}, "
                     f"replan={reflection['replan_recommended']}")

            if reflection["issues"]:
                for issue in reflection["issues"]:
                    self.log(f"  Issue: {issue}")
            if reflection["suggestions"]:
                for sug in reflection["suggestions"]:
                    self.log(f"  Suggestion: {sug}")

            # Save artefacts for this iteration
            save_json(os.path.join(output_dir, "metrics.json"), eval_payload)
            save_json(os.path.join(output_dir, "reflection.json"), reflection)

            # Replan decision
            if not should_replan(reflection):
                self.log("Reflection: no replan needed. Finalising.")
                break

            if self.state["replan_count"] >= self.ctx.max_replans:
                self.log(f"Max replans ({self.ctx.max_replans}) reached. Finalising.")
                break

            self.state["replan_count"] += 1
            self.log(f"Replanning (attempt {self.state['replan_count']})...")
            steps, profile = apply_replan_strategy(steps, profile, reflection)
            plan_dict["steps"] = steps
            plan_dict["hints"]["replan_attempt"] = self.state["replan_count"]
            self.log(f"New plan: {steps}")

            # Save updated plan
            save_json(os.path.join(output_dir, "plan.json"), {
                "steps": steps,
                "rationale": plan_dict["rationale"],
                "hints": plan_dict["hints"],
                "explanation": explain_plan(plan_dict),
                "replan_iteration": self.state["replan_count"],
            })

        # Step 10: Final report and memory update
        write_markdown_report(
            out_path=os.path.join(output_dir, "report.md"),
            ctx=self.ctx,
            fingerprint=fp,
            dataset_profile=profile,
            plan=plan_dict,
            eval_payload=eval_payload,
            reflection=reflection,
        )

        self.memory.upsert_dataset_record(fp, {
            "last_seen": now_iso(),
            "target": self.ctx.target,
            "shape": profile["shape"],
            "best_model": best["model"],
            "best_metrics": best,
            "replan_count": self.state["replan_count"],
        })

        self.log(f"Done. Outputs: {output_dir}")
        return output_dir
