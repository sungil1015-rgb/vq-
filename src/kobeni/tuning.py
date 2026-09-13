from __future__ import annotations

import csv
import json
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from kobeni.config import ExperimentConfig
from kobeni.tracking import mark_run_status
from kobeni.training import train

Direction = Literal["maximize", "minimize"]
ParameterKind = Literal["float", "int", "categorical"]


@dataclass(frozen=True)
class SearchParameter:
    name: str
    target: str
    kind: ParameterKind
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    step: float | int | None = None
    choices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class TuningConfig:
    baseline_config: Path
    study_name: str
    direction: Direction
    metric: str
    output_root: Path
    trial_epochs: int
    trials_per_worker: int
    trial_seed: int
    sampler_seed: int
    sampler_startup_trials: int
    pruner_startup_trials: int
    pruner_warmup_epochs: int
    pruner_interval_epochs: int
    parameters: tuple[SearchParameter, ...]

    @classmethod
    def from_toml(cls, path: str | Path) -> TuningConfig:
        config_path = Path(path)
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        study = raw.get("study", {})
        sampler = raw.get("sampler", {})
        pruner = raw.get("pruner", {})
        parameters = tuple(
            SearchParameter(
                name=name,
                target=value["target"],
                kind=value["type"],
                low=value.get("low"),
                high=value.get("high"),
                log=value.get("log", False),
                step=value.get("step"),
                choices=tuple(value.get("choices", ())),
            )
            for name, value in raw.get("parameters", {}).items()
        )
        config = cls(
            baseline_config=Path(study["baseline_config"]),
            study_name=study["name"],
            direction=study.get("direction", "maximize"),
            metric=study.get("metric", "eval.accuracy"),
            output_root=Path(study.get("output_root", "outputs/optuna")),
            trial_epochs=study.get("trial_epochs", 50),
            trials_per_worker=study.get("trials_per_worker", 8),
            trial_seed=study.get("trial_seed", 0),
            sampler_seed=sampler.get("seed", 2026),
            sampler_startup_trials=sampler.get("startup_trials", 8),
            pruner_startup_trials=pruner.get("startup_trials", 8),
            pruner_warmup_epochs=pruner.get("warmup_epochs", 10),
            pruner_interval_epochs=pruner.get("interval_epochs", 5),
            parameters=parameters,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.study_name.strip():
            raise ValueError("study.name must not be empty")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("study.direction must be maximize or minimize")
        if self.trial_epochs < 1 or self.trials_per_worker < 1:
            raise ValueError("trial_epochs and trials_per_worker must be positive")
        if not self.parameters:
            raise ValueError("At least one tuning parameter is required")
        names: set[str] = set()
        for parameter in self.parameters:
            if parameter.name in names:
                raise ValueError(f"Duplicate parameter: {parameter.name}")
            names.add(parameter.name)
            section, separator, field_name = parameter.target.partition(".")
            if separator != "." or section not in {"model", "data", "train"} or not field_name:
                raise ValueError(f"Invalid parameter target: {parameter.target}")
            if parameter.kind not in {"float", "int", "categorical"}:
                raise ValueError(f"Invalid parameter type: {parameter.kind}")
            if parameter.kind == "categorical":
                if not parameter.choices:
                    raise ValueError(f"{parameter.name} requires choices")
            elif parameter.low is None or parameter.high is None:
                raise ValueError(f"{parameter.name} requires low and high")
            elif parameter.low >= parameter.high:
                raise ValueError(f"{parameter.name} requires low < high")
            if parameter.log and parameter.step is not None:
                raise ValueError(f"{parameter.name} cannot set both log and step")


def _suggest(trial: optuna.Trial, parameter: SearchParameter) -> Any:
    if parameter.kind == "categorical":
        return trial.suggest_categorical(parameter.name, list(parameter.choices))
    if parameter.kind == "int":
        assert isinstance(parameter.low, int) and isinstance(parameter.high, int)
        step = int(parameter.step) if parameter.step is not None else 1
        return trial.suggest_int(
            parameter.name,
            parameter.low,
            parameter.high,
            step=step,
            log=parameter.log,
        )
    assert parameter.low is not None and parameter.high is not None
    return trial.suggest_float(
        parameter.name,
        float(parameter.low),
        float(parameter.high),
        step=float(parameter.step) if parameter.step is not None else None,
        log=parameter.log,
    )


def apply_overrides(
    config: ExperimentConfig,
    tuning_config: TuningConfig,
    values: dict[str, Any],
) -> ExperimentConfig:
    sections: dict[str, dict[str, Any]] = {"model": {}, "data": {}, "train": {}}
    targets = {parameter.name: parameter.target for parameter in tuning_config.parameters}
    for name, value in values.items():
        section, field_name = targets[name].split(".", maxsplit=1)
        if not hasattr(getattr(config, section), field_name):
            raise ValueError(f"Unknown config target: {targets[name]}")
        sections[section][field_name] = value
    updated = replace(
        config,
        model=replace(config.model, **sections["model"]),
        data=replace(config.data, **sections["data"]),
        train=replace(config.train, **sections["train"]),
    )
    updated.validate()
    return updated


def _metric_value(record: dict[str, Any], metric: str) -> float:
    value: Any = record
    for key in metric.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Metric {metric!r} was not found in the epoch record")
        value = value[key]
    if not isinstance(value, int | float):
        raise TypeError(f"Metric {metric!r} is not numeric")
    return float(value)


def _study_directory(config: TuningConfig, run_group: str) -> Path:
    return config.output_root / run_group


def _storage(config: TuningConfig, run_group: str) -> JournalStorage:
    directory = _study_directory(config, run_group)
    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / f"{config.study_name}.journal"
    return JournalStorage(JournalFileBackend(str(journal_path)))


def _load_or_create_study(
    config: TuningConfig,
    run_group: str,
    worker_id: int,
) -> optuna.Study:
    sampler = TPESampler(
        seed=config.sampler_seed + worker_id,
        n_startup_trials=config.sampler_startup_trials,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.pruner_startup_trials,
        n_warmup_steps=config.pruner_warmup_epochs,
        interval_steps=config.pruner_interval_epochs,
    )
    return optuna.create_study(
        storage=_storage(config, run_group),
        sampler=sampler,
        pruner=pruner,
        study_name=config.study_name,
        direction=config.direction,
        load_if_exists=True,
    )


def run_worker(
    tuning_config: TuningConfig,
    run_group: str,
    worker_id: int,
    n_trials: int | None = None,
) -> dict[str, Any]:
    if worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    if n_trials is not None and n_trials < 1:
        raise ValueError("n_trials must be positive")
    study = _load_or_create_study(tuning_config, run_group, worker_id)
    baseline = ExperimentConfig.from_toml(tuning_config.baseline_config)
    study_dir = _study_directory(tuning_config, run_group)

    def objective(trial: optuna.Trial) -> float:
        values = {
            parameter.name: _suggest(trial, parameter) for parameter in tuning_config.parameters
        }
        config = apply_overrides(baseline, tuning_config, values)
        output_dir = study_dir / "trials" / f"trial_{trial.number:05d}"
        config = replace(
            config,
            train=replace(
                config.train,
                epochs=tuning_config.trial_epochs,
                warmup_epochs=min(
                    config.train.warmup_epochs,
                    tuning_config.trial_epochs - 1,
                ),
                seed=tuning_config.trial_seed,
                output_dir=str(output_dir),
                resume_from=None,
            ),
            tracking=replace(
                config.tracking,
                enabled=False,
                run_name=f"trial_{trial.number:05d}",
                run_group=None,
                save_source_snapshot=False,
            ),
        )
        config.validate()
        trial.set_user_attr("worker_id", worker_id)
        trial.set_user_attr("output_dir", str(output_dir))
        trial.set_user_attr("seed", tuning_config.trial_seed)
        best_value: float | None = None

        def report_epoch(record: dict[str, Any]) -> None:
            nonlocal best_value
            value = _metric_value(record, tuning_config.metric)
            if best_value is None:
                best_value = value
            elif tuning_config.direction == "maximize":
                best_value = max(best_value, value)
            else:
                best_value = min(best_value, value)
            trial.report(value, step=int(record["epoch"]))
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at epoch {record['epoch']}: {tuning_config.metric}={value:.6f}"
                )

        try:
            train(config, epoch_callback=report_epoch)
        except optuna.TrialPruned as error:
            mark_run_status(
                output_dir,
                "pruned",
                reason=str(error),
                best_objective_value=best_value,
            )
            raise
        except Exception as error:
            mark_run_status(
                output_dir,
                "failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        if best_value is None:
            raise RuntimeError("Training completed without reporting an objective metric")
        return best_value

    trial_count = n_trials if n_trials is not None else tuning_config.trials_per_worker
    study.optimize(objective, n_trials=trial_count, gc_after_trial=True)
    worker_summary = {
        "worker_id": worker_id,
        "requested_trials": trial_count,
        "study_name": tuning_config.study_name,
        "run_group": run_group,
        "total_study_trials": len(study.trials),
    }
    worker_dir = study_dir / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / f"worker_{worker_id}.json").write_text(
        json.dumps(worker_summary, indent=2), encoding="utf-8"
    )
    return worker_summary


def summarize_study(tuning_config: TuningConfig, run_group: str) -> dict[str, Any]:
    study = optuna.load_study(
        storage=_storage(tuning_config, run_group),
        study_name=tuning_config.study_name,
    )
    study_dir = _study_directory(tuning_config, run_group)
    state_counts: dict[str, int] = {}
    trial_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        state = trial.state.name.lower()
        state_counts[state] = state_counts.get(state, 0) + 1
        trial_rows.append(
            {
                "number": trial.number,
                "state": state,
                "value": trial.value,
                "params": trial.params,
                "worker_id": trial.user_attrs.get("worker_id"),
                "output_dir": trial.user_attrs.get("output_dir"),
            }
        )

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    ranked_trials = sorted(
        completed,
        key=lambda trial: float(trial.value),
        reverse=tuning_config.direction == "maximize",
    )
    best_trial = ranked_trials[0] if ranked_trials else None
    top_trials = [
        {
            "rank": rank,
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "worker_id": trial.user_attrs.get("worker_id"),
            "output_dir": trial.user_attrs.get("output_dir"),
        }
        for rank, trial in enumerate(ranked_trials[:4], start=1)
    ]
    summary: dict[str, Any] = {
        "study_name": tuning_config.study_name,
        "run_group": run_group,
        "direction": tuning_config.direction,
        "metric": tuning_config.metric,
        "trial_epochs": tuning_config.trial_epochs,
        "state_counts": state_counts,
        "total_trials": len(study.trials),
        "best_trial": top_trials[0] if top_trials else None,
        "top_trials": top_trials,
    }
    if best_trial is not None:
        baseline = ExperimentConfig.from_toml(tuning_config.baseline_config)
        top_configs_dir = study_dir / "top_configs"
        top_configs_dir.mkdir(parents=True, exist_ok=True)
        for entry, trial in zip(top_trials, ranked_trials[:4], strict=True):
            rank = entry["rank"]
            rank_dir = top_configs_dir / f"rank_{rank:02d}_trial_{trial.number:05d}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            config = apply_overrides(baseline, tuning_config, trial.params)
            (rank_dir / "config.json").write_text(
                json.dumps(asdict(config), indent=2), encoding="utf-8"
            )
            (rank_dir / "overrides.json").write_text(
                json.dumps(trial.params, indent=2), encoding="utf-8"
            )
            (rank_dir / "trial.json").write_text(
                json.dumps(entry, indent=2), encoding="utf-8"
            )
        best_config = apply_overrides(baseline, tuning_config, best_trial.params)
        (study_dir / "best_config.json").write_text(
            json.dumps(asdict(best_config), indent=2), encoding="utf-8"
        )
        (study_dir / "best_overrides.json").write_text(
            json.dumps(best_trial.params, indent=2), encoding="utf-8"
        )

    (study_dir / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (study_dir / "trials.json").write_text(json.dumps(trial_rows, indent=2), encoding="utf-8")
    parameter_names = [parameter.name for parameter in tuning_config.parameters]
    with (study_dir / "trials.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["number", "state", "value", "worker_id", "output_dir", *parameter_names],
        )
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(
                {
                    "number": row["number"],
                    "state": row["state"],
                    "value": row["value"],
                    "worker_id": row["worker_id"],
                    "output_dir": row["output_dir"],
                    **row["params"],
                }
            )
    return summary
