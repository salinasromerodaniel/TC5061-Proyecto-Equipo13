"""Ejecutar Locust y capturar docker stats simultáneamente, en WSL/Linux."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_compose import verify


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def wait_for_stack(api_url, tracking_uri, payload):
    deadline = time.monotonic() + 120
    while True:
        try:
            return verify(api_url, tracking_uri, payload)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def snapshot(container):
    data = json.loads(command("docker", "inspect", container))[0]
    return {"id": data["Id"], "image_id": data["Image"], "name": data["Name"],
            "image_revision": data["Config"]["Labels"].get("org.opencontainers.image.revision"),
            "state": {key: data["State"][key] for key in ("Status", "OOMKilled", "ExitCode")},
            "restart_count": data["RestartCount"],
            "limits": {key: data["HostConfig"][key] for key in ("NanoCpus", "Memory")}}


class ResourceCapture:
    def __init__(self, containers, output):
        self.containers = containers
        self.output = output
        self.errors = []
        self.stop = threading.Event()

    def __enter__(self):
        self.log = (self.output / "docker-stats.stderr.log").open("w", encoding="utf-8")
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        return self

    def collect(self):
        with (self.output / "docker-stats.jsonl").open("w", encoding="utf-8") as stream:
            while not self.stop.is_set():
                try:
                    sample = subprocess.check_output(
                        ["docker", "stats", "--no-stream", "--format", "{{json .}}",
                         *self.containers], text=True, stderr=self.log, timeout=15)
                    timestamp = time.time()
                    for line in sample.splitlines():
                        record = {"timestamp": timestamp, "stats": json.loads(line)}
                        stream.write(json.dumps(record) + "\n")
                    stream.flush()
                except (ValueError, subprocess.SubprocessError, OSError) as error:
                    self.errors.append(str(error))
                    break
                self.stop.wait(.2)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=20)
        self.log.close()
        if self.thread.is_alive() or self.errors:
            raise RuntimeError(f"Falló la captura de recursos: {self.errors}")


def run_locust(host, users, seconds, output, model):
    output.mkdir()
    env = {**os.environ, "MIBICI_LOAD_MODEL": json.dumps(model)}
    args = [sys.executable, "-m", "locust", "-f", str(ROOT / "load/locustfile.py"),
            "--host", host, "--headless", "--users", str(users),
            "--spawn-rate", str(max(1, users // 5)), "--run-time", f"{seconds}s",
            "--stop-timeout", "15", "--only-summary", "--csv-full-history",
            "--csv", str(output / "locust"), "--html", str(output / "locust.html")]
    started = time.time()
    print(f"Locust: {users} usuarios durante {seconds} s -> {output}", flush=True)
    with (output / "locust.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        generator = psutil.Process(process.pid)
        generator.cpu_percent()
        with (output / "generator.jsonl").open("w", encoding="utf-8") as stream:
            try:
                while process.poll() is None:
                    try:
                        stream.write(json.dumps({"timestamp": time.time(),
                            "cpu_percent": generator.cpu_percent(),
                            "rss_bytes": generator.memory_info().rss}) + "\n")
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(2)
                code = process.wait(timeout=20)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=20)
    if code not in (0, 1):
        raise RuntimeError(f"Locust terminó con código {code}: {output / 'locust.log'}")
    # Código 1 conserva la evidencia de fallos HTTP o validaciones de respuesta.
    if not (output / "locust_stats.csv").is_file():
        raise RuntimeError("Locust no produjo métricas.")
    return {"users": users, "run_time_seconds": seconds, "spawn_rate": max(1, users // 5),
            "started_at_unix": started, "finished_at_unix": time.time(), "exit_code": code}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8014")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5014")
    parser.add_argument("--project", required=True, help="Nombre del proyecto Compose que ya está arriba.")
    parser.add_argument("--users", nargs="+", type=int, default=[1, 10, 50])
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True, help="Directorio nuevo para esta ejecución.")
    args = parser.parse_args()
    if args.seconds < 30 or any(users < 1 for users in args.users):
        parser.error("Usa al menos 30 segundos y usuarios positivos.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload_path = ROOT / "examples/p1-predict.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    initial = wait_for_stack(args.api_url, args.tracking_uri, payload)
    containers = {}
    for service in ("api", "mlflow"):
        container = command("docker", "compose", "-p", args.project, "ps", "-q", service)
        if not container or "\n" in container:
            raise RuntimeError(f"Se requiere un contenedor activo para {service}.")
        containers[service] = container
    before = {key: snapshot(value) for key, value in containers.items()}
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(),
                "code_revision": command("git", "rev-parse", "HEAD"),
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("load/locustfile.py", "scripts/run_load.py",
                                               "scripts/summarize_load.py")},
                "host": {"system": platform.platform(), "python": platform.python_version(),
                         "cpu_model": next((line.split(":", 1)[1].strip() for line in
                             Path("/proc/cpuinfo").read_text().splitlines()
                             if line.startswith("model name")), "unknown"),
                         "logical_cpus": os.cpu_count(), "memory_bytes": psutil.virtual_memory().total,
                         "docker_info": json.loads(command("docker", "info", "--format", "{{json .}}"))["NCPU"]},
                "tools": {"docker": command("docker", "version", "--format", "{{.Server.Version}}"),
                          "compose": command("docker", "compose", "version", "--short"),
                          "packages": command(sys.executable, "-m", "pip", "freeze").splitlines()},
                "api_url": args.api_url, "project": args.project, "preflight": initial,
                "containers_before": before, "batch_size": len(payload["instances"]),
                "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                "protocol": "closed_loop_no_think_time_includes_ramp_5s; generator shares WSL host",
                "scenarios": {}}
    with ResourceCapture(list(containers.values()), output):
        metadata["idle_start"] = time.time()
        time.sleep(20)
        metadata["idle_end"] = time.time()
        run_locust(args.api_url, 2, 15, output / "warmup", initial["health"]["model"])
        for index, users in enumerate(args.users, 1):
            name = f"{index:02d}-users{users}"
            metadata["scenarios"][name] = run_locust(
                args.api_url, users, args.seconds, output / name, initial["health"]["model"])
            time.sleep(5)
    final = verify(args.api_url, args.tracking_uri, payload)
    if final["health"] != initial["health"] or final["prediction"] != initial["prediction"]:
        raise RuntimeError("El modelo o la respuesta cambió durante la prueba.")
    metadata["postflight"] = final
    metadata["containers_after"] = {key: snapshot(value) for key, value in containers.items()}
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Mediciones y reportes Locust guardados en {output}")


if __name__ == "__main__":
    main()
