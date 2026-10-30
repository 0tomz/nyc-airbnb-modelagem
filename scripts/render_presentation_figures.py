"""Render the original APS2 results for reading on the web, without new searches."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
sys.path.insert(0, str(ROOT / "src"))

import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from nyc_airbnb_eda.modeling import build_log_target_pipeline
from nyc_airbnb_eda.preprocessing import split_data

INK, BLUE, CORAL, MUTED = "#17201e", "#4d8392", "#d95f43", "#84908c"
FIGURES = ROOT / "reports/figures/apresentacao"


def finish(fig, name):
    fig.tight_layout(pad=1.5)
    fig.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    results_path = ROOT / "reports/aps2_results.json"
    r = json.loads(results_path.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})

    models = list(r["comparacao"])
    vals = [r["comparacao"][m]["mae_cv"] for m in models]
    stds = [r["comparacao"][m]["desvio_cv"] for m in models]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = [MUTED, BLUE, BLUE, "#e5a18e", CORAL]
    labels = ["Dummy · mediana", "Ridge", "Random Forest", "Ridge + log", "Random Forest + log"]
    ax.barh(labels, vals, color=colors, height=.65)
    ax.errorbar(vals, range(len(vals)), xerr=stds, fmt="none", color=INK, capsize=4)
    for i, (v, sd) in enumerate(zip(vals, stds)):
        ax.text(v + sd + 1.2, i, f"{v:.2f}".replace(".", ","), va="center", fontweight="bold")
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="MAE em dólares · menor é melhor",
           title="A transformação do alvo melhora os dois modelos")
    ax.grid(axis="x", alpha=.12)
    ax.set_axisbelow(True)
    finish(fig, "comparacao_modelos")

    curve = r["curva"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(curve["tamanhos"], curve["mae_treino"], "o-", color=BLUE, label="Treino", linewidth=2.5)
    ax.plot(curve["tamanhos"], curve["mae_validacao"], "o-", color=CORAL, label="Validação cruzada", linewidth=2.5)
    ax.set(xlabel="Anúncios usados no ajuste de cada fold", ylabel="MAE (US$)",
           title="Mais dados reduzem o erro; o hiato de generalização permanece")
    ax.legend(frameon=False)
    ax.grid(alpha=.12)
    finish(fig, "curva_aprendizado")

    for key, name, title, unit in [
        ("floresta", "importancia_floresta", "Tipo de acomodação e coordenadas lideram na floresta", "Importância de impureza · soma = 1"),
        ("shap_global", "shap_global", "Tipo de acomodação tem a maior contribuição no SHAP", "Média de |SHAP| · escala log1p(price)"),
    ]:
        values = r[key]["agregada"] if key == "floresta" else r[key]
        ordered = sorted(values.items(), key=lambda p: p[1], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 5.6))
        names, numbers = zip(*ordered)
        ax.barh(names, numbers, color=[CORAL] + [BLUE] * (len(names) - 1), height=.65)
        ax.invert_yaxis()
        for i, value in enumerate(numbers):
            ax.text(value + max(numbers) * .02, i, f"{value:.3f}".replace(".", ","), va="center", fontsize=11)
        ax.set(xlim=(0, max(numbers) * 1.23), xlabel=unit, title=title)
        ax.grid(axis="x", alpha=.12)
        ax.set_axisbelow(True)
        finish(fig, name)

    top = r["ridge"]["top3"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh([x["feature"] for x in top], [x["coef"] for x in top],
            color=[CORAL if x["coef"] > 0 else BLUE for x in top], height=.55)
    ax.axvline(0, color=INK, linewidth=1)
    ax.invert_yaxis()
    ax.set(xlim=(-.75, .8), xlabel="Coeficiente em log1p(price) · associação, sem efeito causal",
           title="Os três maiores coeficientes em magnitude do Ridge")
    for i, row in enumerate(top):
        v = row["coef"]
        ax.text(v + (.025 if v > 0 else -.025), i, f"{v:+.3f}".replace(".", ","),
                ha="left" if v > 0 else "right", va="center", fontweight="bold")
    finish(fig, "coeficientes_ridge")

    for i, (label, case) in enumerate(r["shap_casos"].items(), 1):
        base = r["shap_base"]
        end = float(np.log1p(case["previsto"]))
        contrib = [float(x["shap"]) for x in case["top"]]
        contrib.append(end - base - sum(contrib))
        def value(v):
            return f"{v:g}" if isinstance(v, (int, float)) else str(v)
        names = [f"{x['variavel']} = {value(x['valor'])}" for x in case["top"]] + ["Demais atributos · soma"]
        fig, ax = plt.subplots(figsize=(11, 4.8))
        running = base
        for j, delta in enumerate(contrib):
            nxt = running + delta
            ax.barh(j, abs(delta), left=min(running, nxt), color=CORAL if delta >= 0 else BLUE, height=.55)
            ax.text(max(running, nxt) + .015, j, f"{delta:+.3f}".replace(".", ","), va="center", fontsize=11)
            if j < len(contrib) - 1:
                ax.plot([nxt, nxt], [j + .28, j + .72], color=MUTED, linestyle=":")
            running = nxt
        assert np.isclose(running, end)
        ax.axvline(base, color=MUTED, linestyle="--", linewidth=1, label=f"Valor-base: {base:.3f}")
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        all_ends = base + np.r_[0, np.cumsum(contrib)]
        ax.set_xlim(min(all_ends) - .06, max(all_ends) + .15)
        titles = ["Maior subestimação", "Maior superestimação", "Erro próximo da mediana"]
        real_label = f"{case['real']:,.0f}".replace(",", ".")
        pred_label = f"{case['previsto']:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        ax.set(title=f"{titles[i-1]} · real US\\$ {real_label} / previsto US\\$ {pred_label}",
               xlabel="Saída do modelo em log1p(price) · laranja eleva; azul reduz")
        ax.legend(loc="lower right", frameon=False, fontsize=10)
        finish(fig, f"shap_caso_{i}")

    # Fit only the published final configuration; no search, resampling or new selection.
    df = pd.read_csv(ROOT / "data/raw/AB_NYC_2019.csv")
    X, test_X, y, test_y = split_data(df)
    params = r["rf_log"]["params"]
    estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=2,
        max_depth=int(params["max_depth"]), min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"])
    with threadpool_limits(limits=2):
        model = build_log_target_pipeline(estimator).fit(X, y)
        pred = model.predict(test_X)
    mae = mean_absolute_error(test_y, pred)
    assert np.isclose(mae, r["teste"]["mae"], atol=1e-6), (mae, r["teste"]["mae"])
    residual = test_y.to_numpy() - pred
    lo, hi = np.quantile(residual, [.01, .99])
    sample = np.random.default_rng(42).choice(len(test_y), size=2500, replace=False)
    p99 = float(np.quantile(test_y, .99))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(test_y.to_numpy()[sample], pred[sample], alpha=.3, s=17, color=BLUE, edgecolors="none")
    ax.plot([0, p99], [0, p99], "--", color=CORAL, linewidth=2, label="Previsão perfeita")
    ax.set(xlim=(0, p99), ylim=(0, p99), xlabel="Preço real (US$)", ylabel="Preço previsto (US$)",
           title="Nos preços altos, as previsões ficam abaixo da diagonal")
    ax.legend(frameon=False)
    finish(fig, "real_previsto")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(pred, residual, alpha=.2, s=12, color=BLUE, edgecolors="none")
    ax.axhline(0, color=CORAL, linestyle="--", linewidth=2)
    ax.set(xlim=(0, np.quantile(pred, .99)), ylim=(lo, hi), xlabel="Preço previsto (US$)",
           ylabel="Resíduo: real − previsto (US$)", title="A dispersão dos erros cresce com o preço previsto")
    finish(fig, "residuos_previsto")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    inside = (residual >= lo) & (residual <= hi)
    ax.hist(residual[inside], bins=60, color=BLUE, edgecolor="white", linewidth=.4)
    ax.axvline(0, color=CORAL, linestyle="--", linewidth=2, label="Erro zero")
    ax.set(xlabel="Resíduo: real − previsto (US$)", ylabel="Anúncios",
           title="A cauda positiva concentra as grandes subestimações")
    ax.legend(frameon=False)
    finish(fig, "distribuicao_residuos")
    errors = pd.DataFrame({"real": test_y.to_numpy(), "erro": np.abs(residual)})
    errors["decil"] = pd.qcut(errors.real, 10, labels=False, duplicates="drop") + 1
    deciles = errors.groupby("decil").erro.mean()
    assert np.allclose(deciles, r["decis"]["mae"])
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([f"D{i}" for i in deciles.index], deciles, color=[BLUE] * 9 + [CORAL], width=.65)
    ax.bar_label(bars, labels=[f"{v:.1f}".replace(".", ",") for v in deciles], padding=4, fontsize=11)
    ax.axhline(mae, color=INK, linestyle="--", linewidth=1, label="MAE geral: US$ 57,53")
    ax.set(ylim=(0, max(deciles) * 1.2), xlabel="Decis do preço real · do mais barato ao mais caro",
           ylabel="MAE no teste (US$)", title="O erro se concentra nos 10% de anúncios mais caros")
    ax.legend(frameon=False, loc="upper left")
    finish(fig, "erro_por_decil")
    checks = {"source_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
              "model": r["final"]["modelo"], "test_n": len(test_y), "mae": float(mae),
              "reference_mae": r["teste"]["mae"], "excluded_from_residual_view": int((~inside).sum()),
              "note": "Presentation only; final configuration and original APS2 results unchanged."}
    (ROOT / "reports/presentation_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(list(FIGURES.glob('*.png')))} figures; original MAE verified: {mae:.8f}")


if __name__ == "__main__":
    main()
