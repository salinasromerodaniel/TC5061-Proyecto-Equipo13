"""Entrada común de MLflow Project y servidor local del equipo."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from mibici import config, dataset


def configured_path(value, default):
    return Path(default if value == "config" else value).resolve()


def read_parameters(value):
    if value == "config":
        return {}
    content = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise dataset.DataError("params_file debe contener un objeto JSON.")
    return content


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="entry", required=True)
    sub.add_parser("server")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--mode", default="config")
    for entry in (ingest, sub.add_parser("train")):
        entry.add_argument("--manifest", default="config")
        entry.add_argument("--data-dir", default="config")
    train = sub.choices["train"]
    train.add_argument("--question", default="config")
    train.add_argument("--model", default="config")
    train.add_argument("--params-file", default="config")
    train.add_argument("--report-dir", default="config")
    args = parser.parse_args(argv)
    if args.entry == "server":
        return subprocess.call([sys.executable, "-m", "mlflow", "server",
                                "--backend-store-uri", "sqlite:///" + str(config.P1_TRACKING_DB),
                                "--host", config.PROJECT_TRACKING_HOST,
                                "--port", str(config.PROJECT_TRACKING_PORT), "--workers", "1"])
    from mibici.tracking import project_run, validate_environment
    from mibici.modeling.p1 import code_commit
    validate_environment()
    commit = code_commit()
    manifest = configured_path(args.manifest, config.PROJECT_DATA_MANIFEST)
    data_dir = configured_path(args.data_dir, config.DATA_DIR)
    if args.entry == "ingest":
        mode = config.PROJECT_DEFAULT_INGEST_MODE if args.mode == "config" else args.mode
        if mode not in ("reproduce", "verify", "repair-stations"):
            raise dataset.DataError("mode debe ser reproduce, verify o repair-stations.")
        with project_run("data", mode, commit, stage="ingest"):
            dataset.main([mode, "--manifest", str(manifest), "--data-dir", str(data_dir)])
            import mlflow
            mlflow.log_artifact(str(manifest), "data")
            mlflow.log_param("data_version", dataset.read_manifest(manifest)["data_version"])
        return 0
    from mibici.modeling.train import dispatch
    question = config.PROJECT_DEFAULT_QUESTION if args.question == "config" else args.question
    if question not in config.PROJECT_DEFAULT_MODELS:
        raise dataset.DataError("question debe ser p1, p2 o p3.")
    model = config.PROJECT_DEFAULT_MODELS[question] if args.model == "config" else args.model
    parameters = read_parameters(args.params_file)
    report_dir = configured_path(args.report_dir, config.PROJECT_REPORT_DIR / question)
    with project_run(question, model, commit) as parent:
        result = dispatch(question, model, parameters, manifest, data_dir, report_dir,
                          parent.info.run_id, commit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (dataset.DataError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
