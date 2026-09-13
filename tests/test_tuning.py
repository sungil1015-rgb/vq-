from dataclasses import replace
from pathlib import Path

import optuna
import pytest

import kobeni.tuning as tuning_module
from kobeni.config import ExperimentConfig
from kobeni.tuning import (
    SearchParameter,
    TuningConfig,
    _metric_value,
    _suggest,
    apply_overrides,
    run_worker,
    summarize_study,
)


def test_load_vq_optuna_config() -> None:
    config = TuningConfig.from_toml("config/tuning/vq_optuna.toml")

    assert config.baseline_config == Path("config/architecture/vq_baseline.toml")
    assert config.direction == "maximize"
    assert config.trial_epochs > 0
    assert config.trials_per_worker > 0
    assert config.parameters


def test_fixed_trial_values_are_applied_to_nested_config() -> None:
    tuning = TuningConfig.from_toml("config/tuning/vq_optuna.toml")
    tuning = replace(
        tuning,
        parameters=(
            SearchParameter(
                name="attention_temperature",
                target="model.attention_temperature",
                kind="float",
                low=2.0,
                high=16.0,
            ),
            SearchParameter(
                name="label_smoothing",
                target="train.label_smoothing",
                kind="float",
                low=0.0,
                high=0.1,
            ),
            SearchParameter(
                name="cutmix_probability",
                target="data.cutmix_probability",
                kind="categorical",
                choices=(0.25, 0.5, 0.75),
            ),
        ),
    )
    baseline = ExperimentConfig.from_toml(tuning.baseline_config)
    values = {
        "attention_temperature": 4.0,
        "label_smoothing": 0.06,
        "cutmix_probability": 0.75,
    }
    trial = optuna.trial.FixedTrial(values)
    suggestions = {parameter.name: _suggest(trial, parameter) for parameter in tuning.parameters}
    config = apply_overrides(baseline, tuning, suggestions)

    assert config.model.attention_temperature == pytest.approx(4.0)
    assert config.train.label_smoothing == pytest.approx(0.06)
    assert config.data.cutmix_probability == pytest.approx(0.75)
    assert config.data.cutmix_probability == pytest.approx(0.75)


def test_metric_value_uses_dotted_path() -> None:
    assert _metric_value({"eval": {"accuracy": 0.42}}, "eval.accuracy") == pytest.approx(0.42)


def test_invalid_target_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[study]
name = "invalid"
baseline_config = "baseline.toml"

[parameters.lr]
target = "unknown.learning_rate"
type = "float"
low = 0.1
high = 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid parameter target"):
        TuningConfig.from_toml(config_path)


def test_worker_records_trial_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tuning = replace(
        TuningConfig.from_toml("config/tuning/vq_optuna.toml"),
        output_root=tmp_path,
        trial_epochs=2,
    )

    train_calls = 0

    def fake_train(config: ExperimentConfig, epoch_callback=None) -> dict[str, object]:
        nonlocal train_calls
        assert epoch_callback is not None
        train_calls += 1
        score = train_calls / 10
        epoch_callback({"epoch": 1, "eval": {"accuracy": score - 0.01}})
        epoch_callback({"epoch": 2, "eval": {"accuracy": score}})
        return {"best_eval_accuracy": score}

    monkeypatch.setattr(tuning_module, "train", fake_train)
    worker = run_worker(tuning, "test_group", worker_id=0, n_trials=4)
    summary = summarize_study(tuning, "test_group")

    assert worker["requested_trials"] == 4
    assert summary["state_counts"] == {"complete": 4}
    assert summary["best_trial"]["value"] == pytest.approx(0.4)
    assert [entry["rank"] for entry in summary["top_trials"]] == [1, 2, 3, 4]
    assert [entry["value"] for entry in summary["top_trials"]] == pytest.approx(
        [0.4, 0.3, 0.2, 0.1]
    )
    assert (tmp_path / "test_group/study_summary.json").is_file()
    assert (tmp_path / "test_group/trials.csv").is_file()
    assert (tmp_path / "test_group/best_config.json").is_file()
    assert len(list((tmp_path / "test_group/top_configs").glob("rank_*"))) == 4
