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

    fig, ax = plt.subplots(figsize=(6.25, 8))
    ax.set(xlim=(0, 6.25), ylim=(0, 8))
    ax.axis("off")
    nodes = {
        "csv": (0.15, 7.05, "mibici\nCSV de viajes y estaciones"),
        "ingest": (3.45, 7.05, "mibici.dataset\nLimpieza y SHA-256"),
        "data": (3.45, 5.95, "data/processed\nParquet local"),
        "features": (3.45, 4.85, "mibici.features\nSemanas y retardos"),
        "others": (0.15, 2.65, "p2 / p3\nIntegración prevista"),
        "inegi": (0.15, 4.85, "inegi\nGeoJSON de Zapopan"),
        "geo": (0.15, 3.75, "mibici.geography\nPunto en polígono"),
        "manifest": (0.15, 5.95, "references\nManifiestos en Git"),
        "p1": (3.45, 3.75, "mibici.modeling.p1\nBaselines y evaluación"),
        "tracking": (3.45, 2.65, "mlflow\nSQLite y artefactos"),
        "reports": (3.45, 1.55, "reports\nMétricas y figuras"),
    }
    width, height = 2.6, 0.75
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
    arrow((2.85, 7.425), (3.35, 7.425))
    for y in (7.05, 5.95, 4.85, 3.75):
        arrow((4.75, y - .05), (4.75, y - .30))
    ax.plot([3.4, 3.2, 3.2], [4.125, 4.125, 1.925], color="#455a64", linewidth=1)
    arrow((3.2, 1.925), (3.4, 1.925))
    arrow((3.35, 7.15), (2.85, 6.55))
    arrow((1.45, 4.80), (1.45, 4.55))
    arrow((2.85, 4.15), (3.35, 5.15))
    ax.plot([3.35, 3.05, 3.05], [4.95, 4.75, 3.025], color="#455a64", linestyle="--", linewidth=1)
    arrow((3.05, 3.025), (2.85, 3.025), True)
    ax.set_title("Equipo 13 · Arquitectura de P1", fontsize=12, pad=8)
    ax.text(.15, .72, "Datos, modelos y tracking: locales.\nGit: código, documentos, metadatos y evidencias.", fontsize=8.5)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(config.FIGURES_DIR / f"arquitectura.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture_figure()
