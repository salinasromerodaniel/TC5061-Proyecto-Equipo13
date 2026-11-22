"""Tablas y figuras reproducibles a partir de CSV Locust y docker stats originales."""
import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def memory_mib(text):
    value, unit = re.fullmatch(r"([\d.]+)\s*([A-Za-z]+)", text.strip()).groups()
    factors = {"B": 1, "kB": 1000, "MB": 1000**2, "GB": 1000**3,
               "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    return float(value) * factors[unit] / 1024**2


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    metadata = json.loads((root / "metadata.json").read_text())
    records = [json.loads(line) for line in (root / "docker-stats.jsonl").read_text().splitlines()]
    resources = []
    for record in records:
        stats = record["stats"]
        service = next(key for key, value in metadata["containers_before"].items()
                       if value["id"].startswith(stats["ID"]))
        used, limit = stats["MemUsage"].split("/")
        resources.append({"timestamp": record["timestamp"], "service": service,
                          "cpu_percent": float(stats["CPUPerc"].rstrip("%")),
                          "memory_mib": memory_mib(used), "limit_mib": memory_mib(limit)})
    write_csv(root / "resources.csv", resources)
    summaries = []
    resource_summary = []
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for name, spec in metadata["scenarios"].items():
        stats = next(row for row in read_csv(root / name / "locust_stats.csv")
                     if row["Name"] == "Aggregated")
        count, fails = int(stats["Request Count"]), int(stats["Failure Count"])
        if count <= 0:
            raise ValueError(f"No hubo peticiones en {name}.")
        requests_s = float(stats["Requests/s"])
        summaries.append({"scenario": name, "users": spec["users"], "requests": count,
                          "failures": fails, "error_percent": 100 * fails / count,
                          "requests_per_second": requests_s,
                          "successful_inferences_per_second": requests_s * (1 - fails / count)
                                                                    * metadata["batch_size"],
                          "mean_ms": float(stats["Average Response Time"]),
                          "p50_ms": float(stats["50%"]), "p95_ms": float(stats["95%"]),
                          "p99_ms": float(stats["99%"]),
                          "max_ms": float(stats["Max Response Time"])})
        history = [row for row in read_csv(root / name / "locust_stats_history.csv")
                   if row["Name"] == "Aggregated" and int(row["Total Request Count"]) > 0]
        start = float(history[0]["Timestamp"])
        times = [float(row["Timestamp"]) - start for row in history]
        label = f"{spec['users']} usuarios"
        axes[0].plot(times, [float(row["Requests/s"]) for row in history], label=label)
        axes[1].plot(times, [float(row["95%"] or 0) for row in history], label=label)
        axes[2].plot(times, [100 * float(row["Failures/s"]) / float(row["Requests/s"])
                           if float(row["Requests/s"]) else 0 for row in history], label=label)
    axes[0].set(ylabel="Peticiones / s", title="POST /predict · dos inferencias por petición")
    axes[1].set(ylabel="Latencia p95 (ms)")
    axes[2].set(ylabel="Fallos (%)", xlabel="Segundos desde la primera muestra Locust")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(root / "performance.png", dpi=160)
    plt.close(fig)

    intervals = {"idle": (metadata["idle_start"], metadata["idle_end"]),
                 **{name: (spec["started_at_unix"], spec["finished_at_unix"])
                    for name, spec in metadata["scenarios"].items()}}
    for name, (start, end) in intervals.items():
        for service in ("api", "mlflow"):
            selected = [row for row in resources
                        if row["service"] == service and start <= row["timestamp"] <= end]
            if len(selected) < 3:
                raise ValueError(f"Faltan muestras de {service} en {name}.")
            resource_summary.append({"scenario": name, "service": service, "samples": len(selected),
                                     "cpu_mean_percent": mean(row["cpu_percent"] for row in selected),
                                     "cpu_max_percent": max(row["cpu_percent"] for row in selected),
                                     "memory_mean_mib": mean(row["memory_mib"] for row in selected),
                                     "memory_max_mib": max(row["memory_mib"] for row in selected),
                                     "memory_limit_mib": selected[0]["limit_mib"]})
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    start = metadata["idle_start"]
    for service in ("api", "mlflow"):
        selected = [row for row in resources if row["service"] == service]
        times = [row["timestamp"] - start for row in selected]
        axes[0].plot(times, [row["cpu_percent"] for row in selected], label=service)
        axes[1].plot(times, [row["memory_mib"] for row in selected], label=service)
    for name, spec in metadata["scenarios"].items():
        for ax in axes:
            ax.axvspan(spec["started_at_unix"] - start, spec["finished_at_unix"] - start,
                       color="grey", alpha=.1)
        axes[0].text(spec["started_at_unix"] - start, axes[0].get_ylim()[1] * .9,
                     f"{spec['users']} usuarios", fontsize=9)
    axes[0].axhline(100, color="black", linestyle="--", alpha=.4, label="1 CPU")
    axes[0].set(ylabel="CPU (%)", title="Recursos observados con docker stats")
    axes[1].set(ylabel="Memoria (MiB)", xlabel="Segundos desde el inicio del reposo")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(root / "resources.png", dpi=160)
    plt.close(fig)
    generators = []
    for name in metadata["scenarios"]:
        samples = [json.loads(line) for line in (root / name / "generator.jsonl").read_text().splitlines()]
        generators.append({"scenario": name, "cpu_mean_percent": mean(s["cpu_percent"] for s in samples),
                           "cpu_max_percent": max(s["cpu_percent"] for s in samples),
                           "rss_max_mib": max(s["rss_bytes"] for s in samples) / 1024**2})
    write_csv(root / "summary.csv", summaries)
    write_csv(root / "resource-summary.csv", resource_summary)
    write_csv(root / "generator-summary.csv", generators)
    print(json.dumps({"load": summaries, "resources": resource_summary, "generator": generators}, indent=2))


if __name__ == "__main__":
    main()
