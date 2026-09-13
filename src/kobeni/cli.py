from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from kobeni.config import ExperimentConfig
from kobeni.data import resolve_image_shape
from kobeni.models import SpatialVocabularyModel, SpatialVocabularySegmentationModel


def _check(config: ExperimentConfig) -> None:
    height, width = resolve_image_shape(config.data)
    images = torch.randn(2, 3, height, width)
    if config.model.task == "classification":
        model = SpatialVocabularyModel(config.model)
        targets = torch.randint(0, config.model.num_classes, (2,))
        output = model(images)
        from kobeni.training import compute_loss

        loss, components = compute_loss(
            output,
            images,
            targets,
            lambda_vq=config.train.lambda_vq,
            lambda_rec=config.train.lambda_rec,
            lambda_cls=config.train.lambda_cls,
            label_smoothing=config.train.label_smoothing,
            reconstruction_loss=config.train.reconstruction_loss,
            classification_loss=config.train.classification_loss,
        )
        supervision_shape = list(targets.shape)
    else:
        model = SpatialVocabularySegmentationModel(config.model)
        targets = torch.randint(0, config.model.num_classes, (2, height, width))
        output = model(images)
        from kobeni.segmentation_training import compute_segmentation_loss

        loss, components, supervision = compute_segmentation_loss(
            output,
            images,
            targets,
            config.train.lambda_cls,
            config.train.lambda_vq,
            config.train.lambda_rec,
            config.train.label_smoothing,
            config.data.segmentation_ignore_index,
        )
        supervision_shape = list(supervision.shape)
    loss.backward()
    report = {
        "task": config.model.task,
        "variant": config.model.variant,
        "encoder_blocks": config.model.encoder_blocks,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "logits_shape": list(output.logits.shape),
        "latent_shape": list(output.latent.shape),
        "indices_shape": list(output.indices.shape) if output.indices is not None else None,
        "supervision_shape": supervision_shape,
        "loss": loss.item(),
        "loss_components": {name: value.item() for name, value in components.items()},
        "codebook_metrics": {name: value.item() for name, value in output.codebook_metrics.items()},
    }
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kobeni")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--config", type=Path, required=True)
    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("--config", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--output-dir", type=Path)
    train_parser.add_argument("--resume-from", type=Path)
    train_parser.add_argument("--run-name")
    train_parser.add_argument("--run-group")
    usage_parser = subparsers.add_parser("analyze-code-usage")
    usage_parser.add_argument("--config", type=Path, required=True)
    usage_parser.add_argument("--checkpoint", type=Path, required=True)
    usage_parser.add_argument("--split", choices=("train", "test"), default="test")
    usage_parser.add_argument("--output-dir", type=Path, required=True)
    accuracy_parser = subparsers.add_parser("summarize-accuracy")
    accuracy_parser.add_argument("--run-dir", type=Path, required=True)
    experiments_parser = subparsers.add_parser("summarize-experiments")
    experiments_parser.add_argument("--run-dir", type=Path, required=True)
    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--study-config", type=Path, required=True)
    tune_parser.add_argument("--run-group", required=True)
    tune_parser.add_argument("--worker-id", type=int, required=True)
    tune_parser.add_argument("--n-trials", type=int)
    tuning_summary_parser = subparsers.add_parser("summarize-tuning")
    tuning_summary_parser.add_argument("--study-config", type=Path, required=True)
    tuning_summary_parser.add_argument("--run-group", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "summarize-accuracy":
        from kobeni.analysis.accuracy import summarize_accuracy

        print(json.dumps(summarize_accuracy(args.run_dir), indent=2))
        return

    if args.command == "summarize-experiments":
        from kobeni.analysis.experiments import summarize_experiments

        print(json.dumps(summarize_experiments(args.run_dir), indent=2))
        return

    if args.command in {"tune", "summarize-tuning"}:
        from kobeni.tuning import TuningConfig, run_worker, summarize_study

        tuning_config = TuningConfig.from_toml(args.study_config)
        if args.command == "tune":
            result = run_worker(
                tuning_config,
                args.run_group,
                args.worker_id,
                args.n_trials,
            )
        else:
            result = summarize_study(tuning_config, args.run_group)
        print(json.dumps(result, indent=2))
        return

    config = ExperimentConfig.from_toml(args.config)
    if args.command == "prepare-data":
        if config.data.dataset == "ade20k":
            from kobeni.ade20k import prepare_ade20k_data

            prepare_ade20k_data(config.data)
        else:
            from kobeni.data import prepare_cifar_data

            prepare_cifar_data(config.data)
        print(json.dumps({"dataset": config.data.dataset, "root": config.data.root}))
    elif args.command == "check":
        _check(config)
    elif args.command == "train":
        from kobeni.training import train

        train_overrides = {}
        if args.seed is not None:
            train_overrides["seed"] = args.seed
        if args.output_dir is not None:
            train_overrides["output_dir"] = str(args.output_dir)
        if args.resume_from is not None:
            train_overrides["resume_from"] = str(args.resume_from)
        if train_overrides:
            config = replace(
                config,
                train=replace(config.train, **train_overrides),
            )
        tracking_overrides = {}
        if args.run_name is not None:
            tracking_overrides["run_name"] = args.run_name
        if args.run_group is not None:
            tracking_overrides["run_group"] = args.run_group
        if tracking_overrides:
            config = replace(
                config,
                tracking=replace(config.tracking, **tracking_overrides),
            )
        config.validate()
        train(config)
    elif args.command == "analyze-code-usage":
        from kobeni.analysis.code_usage import analyze_code_usage

        analyze_code_usage(config, args.checkpoint, args.split, args.output_dir)
