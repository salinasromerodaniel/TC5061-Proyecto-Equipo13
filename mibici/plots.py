"""Diagrama reproducible de la base de datos y del pipeline implementado de P1."""

import os

from mibici import config


def architecture_figure():
    """Mostrar responsables, datos locales y extensión prevista hacia P2/P3."""
    os.environ.setdefault("MPLCONFIGDIR", str(config.DATA_DIR / "cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set(xlim=(0, 15), ylim=(0, 6))
    ax.axis("off")
    nodes = {
        "csv": (0.3, 4.2, "mibici\nCSV de viajes\ny estaciones"),
        "ingest": (3.2, 4.2, "mibici.dataset\nLimpieza, tipos\ny SHA-256"),
        "data": (6.1, 4.2, "data/processed\nParquet local\nFuera de Git"),
        "features": (9.0, 4.2, "mibici.features\nSemanas completas\ny variables históricas"),
        "others": (11.9, 4.2, "p2 / p3\nIntegración prevista\nOtros responsables"),
        "inegi": (0.3, 1.8, "inegi\nPolígono de Zapopan\nGeoJSON 14120"),
        "geo": (3.2, 1.8, "mibici.geography\nCoordenadas dentro\ndel polígono"),
        "manifest": (6.1, 1.8, "references\nFuentes, conteos\ny hashes en Git"),
        "p1": (9.0, 1.8, "mibici.modeling.p1\nBaselines semanales\nEvaluación temporal"),
        "tracking": (11.9, 1.8, "mlflow\nSQLite, parámetros\ny artefactos locales"),
    }
    width, height = 2.5, 1.05
    for name, (x, y, text) in nodes.items():
        pending = name == "others"
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08",
                             facecolor="#f1f3f5" if pending else "#eaf3fa",
                             edgecolor="#6b7280" if pending else "#1e5275",
                             linestyle="--" if pending else "-", linewidth=1.3)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, ha="center", va="center", fontsize=9)
    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     color="#455a64", linestyle="--" if dashed else "-"))
    for source, target in [("csv", "ingest"), ("ingest", "data"), ("data", "features"),
                           ("features", "others"), ("inegi", "geo"), ("p1", "tracking")]:
        x, y, _ = nodes[source]; u, v, _ = nodes[target]
        arrow((x + width + .08, y + height/2), (u-.08, v+height/2), target == "others")
    arrow((10.25, 4.1), (10.25, 2.95))
    arrow((4.45, 4.1), (7.35, 2.95))
    arrow((5.78, 2.5), (8.92, 4.42))
    ax.text(7.6, 0.85, "Salidas de P1: reports/p1 (métricas) y reports/figures/p1 (gráficas)",
            ha="center", fontsize=10)
    arrow((10.25, 1.7), (10.25, 1.15))
    ax.set_title("Equipo 13 · Base reproducible y arquitectura de P1", fontsize=16, pad=12)
    ax.text(0.3, 0.25, "Datos, modelos y tracking permanecen locales. Git conserva código, documentación, metadatos y evidencias.", fontsize=9)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"arquitectura.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture_figure()
