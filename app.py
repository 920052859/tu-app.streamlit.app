# -*- coding: utf-8 -*-
"""
============================================================================
 Panini Smart Collector: Simulador de gasto, probabilidad e intercambio
============================================================================

Aplicación web interactiva (Streamlit) que estima el costo y la probabilidad
de completar un álbum de figuritas (cromos) tipo Panini, inspirada en el
"Panini Collector Problem".

Base de referencia (álbum Qatar 2022):
    - 980 figuritas distintas
    - 5 figuritas por sobre
    - Sin intercambio: ~899 sobres para completar
    - Con 10 coleccionistas que intercambian: ~321 sobres

Enfoques matemáticos usados:
    1. Coupon Collector Problem  -> estimación analítica base.
    2. Simulación Monte Carlo    -> distribución de escenarios posibles.
    3. Modelo de intercambio      -> reducción de sobres esperados según
                                     repetidas, número de personas y
                                     figuritas ya obtenidas por intercambio
                                     (aproximación "double dixie cup" /
                                     Newman-Shepp para g coleccionistas).

Ejecución:
    pip install streamlit numpy pandas plotly
    streamlit run app.py

Referencia:
    Rodrigo Gonzalez & Carlos A. Catania, "The Panini collector problem".
    https://rpubs.com/rodralez/panini
============================================================================
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Constante de Euler-Mascheroni (aparece en la aproximación del número armónico).
EULER_GAMMA = 0.5772156649015329


# ===========================================================================
# 1) MODELO ANALÍTICO:  Coupon Collector Problem
# ===========================================================================
def harmonic_number(k: int) -> float:
    """
    Devuelve el k-ésimo número armónico  H_k = 1 + 1/2 + ... + 1/k.

    Para valores grandes usa la aproximación asintótica
        H_k ~= ln(k) + gamma + 1/(2k)
    que es muy precisa y evita sumar miles de términos.
    """
    if k <= 0:
        return 0.0
    if k <= 2000:
        # Suma exacta para valores moderados.
        return float(np.sum(1.0 / np.arange(1, k + 1)))
    # Aproximación asintótica para k grande.
    return math.log(k) + EULER_GAMMA + 1.0 / (2.0 * k)


def coupon_collector_estimate(n_total: int, missing: int, per_pack: int):
    """
    Estimación analítica (Coupon Collector) del esfuerzo para conseguir
    `missing` figuritas NUEVAS distintas, partiendo de un álbum con
    (n_total - missing) figuritas ya pegadas.

    Teoría
    ------
    Si tengo `c` figuritas distintas, la probabilidad de que una figurita
    aleatoria sea nueva es (n - c)/n, por lo que en promedio necesito
    n/(n - c) figuritas para conseguir una nueva. Sumando desde c hasta
    completar:

        E[figuritas] = n * (1/1 + 1/2 + ... + 1/missing) = n * H_missing

    Parameters
    ----------
    n_total : int   Total de figuritas distintas del álbum.
    missing : int   Figuritas que faltan.
    per_pack: int   Figuritas por sobre.

    Returns
    -------
    (sobres_esperados, figuritas_esperadas)
    """
    if missing <= 0 or n_total <= 0:
        return 0.0, 0.0
    missing = min(missing, n_total)
    expected_stickers = n_total * harmonic_number(missing)
    expected_packs = expected_stickers / max(per_pack, 1)
    return expected_packs, expected_stickers


# ===========================================================================
# 2) SIMULACIÓN MONTE CARLO
# ===========================================================================
@st.cache_data(show_spinner=False)
def simulate_panini_completion(n_total: int, missing: int, per_pack: int,
                               n_sims: int, max_packs: int = 20000,
                               seed: int = 42) -> np.ndarray:
    """
    Simula `n_sims` realizaciones del proceso de completar el álbum comprando
    sobres, y devuelve cuántos sobres necesitó cada simulación.

    Cada sobre contiene `per_pack` figuritas elegidas de forma uniforme y con
    reemplazo entre las `n_total` figuritas posibles. Se asume (por simetría)
    que las (n_total - missing) figuritas ya poseídas son las primeras índices.

    La simulación está vectorizada con NumPy: todas las realizaciones avanzan
    en paralelo, sobre a sobre, hasta completarse.

    Returns
    -------
    np.ndarray de enteros con el número de sobres por simulación.
    """
    missing = min(missing, n_total)
    if missing <= 0:
        return np.zeros(n_sims, dtype=int)

    rng = np.random.default_rng(seed)
    owned = n_total - missing

    # Matriz [n_sims x n_total] de figuritas conseguidas (True = la tengo).
    collected = np.zeros((n_sims, n_total), dtype=bool)
    if owned > 0:
        collected[:, :owned] = True

    distinct = np.full(n_sims, owned, dtype=int)   # figuritas distintas por sim
    packs_needed = np.zeros(n_sims, dtype=int)      # contador de sobres por sim
    active = distinct < n_total                     # simulaciones aún incompletas

    p = 0
    while active.any() and p < max_packs:
        p += 1
        idx = np.where(active)[0]                    # índices de sims activas
        # Sobres = per_pack figuritas aleatorias por cada sim activa.
        draws = rng.integers(0, n_total, size=(idx.size, per_pack))
        rows = np.repeat(idx, per_pack)
        cols = draws.reshape(-1)
        collected[rows, cols] = True                 # pego las nuevas
        packs_needed[idx] += 1
        # Recalculo figuritas distintas solo para las sims activas.
        distinct[idx] = collected[idx].sum(axis=1)
        active = distinct < n_total

    return packs_needed


# ===========================================================================
# 3) MODELO DE INTERCAMBIO  (reducción de sobres esperados)
# ===========================================================================
def _collaborative_factor(n_total: int, n_people: int) -> float:
    """
    Factor de reducción de sobres por coleccionista cuando `n_people`
    coleccionistas intercambian libremente sus repetidas.

    Se basa en el problema "double dixie cup" (Newman-Shepp): el número
    esperado de figuritas para que un grupo de g coleccionistas complete
    TODOS sus álbumes crece como

        E_g ~= n * ( ln(n) + (g-1)*ln(ln(n)) + gamma )

    El costo POR PERSONA respecto a coleccionar en solitario es:

        r(g) = E_g / ( g * E_1 ),    con  E_1 = n*(ln n + gamma)

    Para n=638:  r(1)=1.0   y   r(10) ~= 0.34  (coincide con el estudio:
    de ~899 sobres en solitario a ~300 con 10 personas).
    """
    if n_people <= 1:
        return 1.0
    ln_n = math.log(n_total)
    e1 = ln_n + EULER_GAMMA
    eg = ln_n + (n_people - 1) * math.log(ln_n) + EULER_GAMMA
    r = eg / (n_people * e1)
    # Acotamos el factor a un rango razonable.
    return float(min(max(r, 0.05), 1.0))


def estimate_exchange_savings(n_total: int, missing: int, duplicates: int,
                              n_people: int, exchanged_already: int,
                              per_pack: int, price: float):
    """
    Estima sobres y gasto CON intercambio, y el ahorro frente a comprar solo.

    El modelo combina dos efectos:
      (a) Colaboración: con más coleccionistas, cada uno necesita comprar
          menos sobres (factor `_collaborative_factor`).
      (b) Ventaja inicial: las figuritas repetidas y las ya obtenidas por
          intercambio se canjean por figuritas que faltan, reduciendo el
          número efectivo de figuritas pendientes antes de comprar.

    Returns
    -------
    dict con: packs_solo, cost_solo, packs_exchange, cost_exchange,
              savings, savings_pct, reduction_factor
    """
    packs_solo, _ = coupon_collector_estimate(n_total, missing, per_pack)
    cost_solo = packs_solo * price

    # Sin personas para intercambiar -> no hay ahorro por colaboración,
    # pero las repetidas y lo ya intercambiado sí dan una ventaja inicial.
    reduction = _collaborative_factor(n_total, n_people)

    # Eficiencia de canje: con más socios es más fácil colocar repetidas.
    trade_eff = n_people / (n_people + 1) if n_people > 0 else 0.0
    head_start = min(duplicates * trade_eff + exchanged_already, missing)
    missing_eff = max(missing - head_start, 0)

    packs_eff, _ = coupon_collector_estimate(n_total, int(round(missing_eff)),
                                             per_pack)
    packs_exchange = packs_eff * reduction
    cost_exchange = packs_exchange * price

    savings = max(cost_solo - cost_exchange, 0.0)
    savings_pct = (savings / cost_solo * 100.0) if cost_solo > 0 else 0.0

    return {
        "packs_solo": packs_solo,
        "cost_solo": cost_solo,
        "packs_exchange": packs_exchange,
        "cost_exchange": cost_exchange,
        "savings": savings,
        "savings_pct": savings_pct,
        "reduction_factor": reduction,
    }


# ===========================================================================
# 4) PROBABILIDAD DE COMPLETAR SEGÚN PRESUPUESTO
# ===========================================================================
def calculate_probability_completion(sim_packs: np.ndarray, budget: float,
                                     price: float) -> float:
    """
    Probabilidad (empírica, vía Monte Carlo) de completar el álbum dado un
    presupuesto. Cuenta qué fracción de simulaciones necesitó un número de
    sobres comprable con el presupuesto.

        sobres_comprables = budget / price
        P(completar) = fracción de sims con packs_needed <= sobres_comprables
    """
    if price <= 0 or len(sim_packs) == 0:
        return 0.0
    affordable_packs = budget / price
    return float(np.mean(sim_packs <= affordable_packs))


# ===========================================================================
#   COMPONENTES VISUALES (Plotly)
# ===========================================================================
PLOT_TEMPLATE = "plotly_white"
COLOR_SOLO = "#ef4444"       # rojo  -> sin intercambio
COLOR_EXCHANGE = "#22c55e"   # verde -> con intercambio
COLOR_ACCENT = "#6366f1"     # índigo


def plot_cost_curve(n_total, per_pack, price):
    """Gráfico 1: gasto estimado (Coupon Collector) según figuritas faltantes."""
    xs = np.arange(1, n_total + 1)
    ys = [coupon_collector_estimate(n_total, int(m), per_pack)[0] * price
          for m in xs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color=COLOR_ACCENT, width=3),
                             name="Gasto estimado",
                             hovertemplate="Faltan %{x} figuritas<br>"
                                           "Gasto ~ S/ %{y:,.0f}<extra></extra>"))
    fig.update_layout(template=PLOT_TEMPLATE,
                      title="Curva de gasto estimado según figuritas faltantes",
                      xaxis_title="Figuritas faltantes",
                      yaxis_title="Gasto estimado (S/)",
                      height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_cost_comparison(cost_solo, cost_exchange):
    """Gráfico 2: comparación de gasto con y sin intercambio."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Sin intercambio", "Con intercambio"],
        y=[cost_solo, cost_exchange],
        marker_color=[COLOR_SOLO, COLOR_EXCHANGE],
        text=[f"S/ {cost_solo:,.0f}", f"S/ {cost_exchange:,.0f}"],
        textposition="outside",
    ))
    fig.update_layout(template=PLOT_TEMPLATE,
                      title="Comparación de gasto: con vs. sin intercambio",
                      yaxis_title="Gasto estimado (S/)",
                      height=380, margin=dict(l=10, r=10, t=50, b=10),
                      showlegend=False)
    return fig


def plot_monte_carlo(sim_packs, affordable_packs=None):
    """Gráfico 3: distribución Monte Carlo de sobres necesarios."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sim_packs, nbinsx=40,
                               marker_color=COLOR_ACCENT, opacity=0.85,
                               name="Simulaciones"))
    mean_packs = float(np.mean(sim_packs)) if len(sim_packs) else 0
    fig.add_vline(x=mean_packs, line_dash="dash", line_color=COLOR_SOLO,
                  annotation_text=f"Media: {mean_packs:,.0f}",
                  annotation_position="top")
    if affordable_packs is not None:
        fig.add_vline(x=affordable_packs, line_dash="dot",
                      line_color=COLOR_EXCHANGE,
                      annotation_text=f"Presupuesto: {affordable_packs:,.0f}",
                      annotation_position="bottom")
    fig.update_layout(template=PLOT_TEMPLATE,
                      title="Distribución Monte Carlo de sobres necesarios",
                      xaxis_title="Sobres necesarios para completar",
                      yaxis_title="Frecuencia (simulaciones)",
                      height=380, margin=dict(l=10, r=10, t=50, b=10),
                      showlegend=False)
    return fig


def plot_savings_curve(n_total, missing, duplicates, exchanged_already,
                       per_pack, price, max_people=20):
    """Gráfico 4: ahorro estimado por intercambio según número de personas."""
    xs = np.arange(0, max_people + 1)
    ys = []
    for g in xs:
        res = estimate_exchange_savings(n_total, missing, duplicates,
                                        int(g), exchanged_already,
                                        per_pack, price)
        ys.append(res["savings"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                             line=dict(color=COLOR_EXCHANGE, width=3),
                             fill="tozeroy",
                             fillcolor="rgba(34,197,94,0.15)",
                             name="Ahorro",
                             hovertemplate="%{x} personas<br>"
                                           "Ahorro ~ S/ %{y:,.0f}<extra></extra>"))
    fig.update_layout(template=PLOT_TEMPLATE,
                      title="Ahorro acumulado por intercambio según Nº de personas",
                      xaxis_title="Personas con las que intercambio",
                      yaxis_title="Ahorro estimado (S/)",
                      height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ===========================================================================
#   ESTILOS  (CSS para una interfaz moderna)
# ===========================================================================
CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; }
.big-title {
    font-size: 2.5rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(90deg,#6366f1,#22c55e);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: .2rem;
}
.subtitle { color:#64748b; font-size:1.05rem; margin-bottom:1.2rem; }
.kpi-card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
    padding:1rem 1.2rem; box-shadow:0 2px 8px rgba(15,23,42,.05);
    height:100%;
}
.kpi-label { color:#64748b; font-size:.82rem; font-weight:600;
             text-transform:uppercase; letter-spacing:.04em; }
.kpi-value { font-size:1.6rem; font-weight:800; color:#0f172a; margin-top:.2rem; }
.kpi-sub  { color:#94a3b8; font-size:.78rem; }
.msg-box {
    border-radius:14px; padding:.9rem 1.1rem; margin:.4rem 0;
    font-weight:600; border:1px solid transparent;
}
.msg-green { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
.msg-blue  { background:#eff6ff; color:#1e40af; border-color:#bfdbfe; }
.msg-amber { background:#fffbeb; color:#92400e; border-color:#fde68a; }
</style>
"""


def kpi_card(label, value, sub=""):
    """Renderiza una tarjeta KPI con HTML."""
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def message_box(text, kind="blue"):
    """Renderiza un mensaje interpretativo con color según el tipo."""
    st.markdown(f'<div class="msg-box msg-{kind}">{text}</div>',
                unsafe_allow_html=True)


# ===========================================================================
#   APLICACIÓN PRINCIPAL
# ===========================================================================
def main():
    st.set_page_config(page_title="Panini Smart Collector",
                       page_icon="⚽", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ---------------------- Encabezado ----------------------
    st.markdown('<div class="big-title">⚽ Panini Smart Collector</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Simulador de gasto, probabilidad e '
                'intercambio · basado en el <i>Panini Collector Problem</i>'
                '</div>', unsafe_allow_html=True)

    # ---------------------- Sidebar (inputs) ----------------------
    with st.sidebar:
        st.header("⚙️ Parámetros")

        st.subheader("📒 Álbum")
        n_total = st.number_input("1. Total de figuritas del álbum",
                                  min_value=1, value=638, step=1)
        have = st.number_input("2. Figuritas que ya tengo",
                               min_value=0, value=458, step=1)
        missing = st.number_input("3. Figuritas que me faltan",
                                  min_value=0, value=int(n_total - 458),
                                  step=1)
        duplicates = st.number_input("4. Figuritas repetidas",
                                     min_value=0, value=320, step=1)

        st.subheader("💸 Compra")
        price = st.number_input("5. Precio por sobre (S/)",
                                min_value=0.1, value=1.50, step=0.10,
                                format="%.2f")
        per_pack = st.number_input("6. Figuritas por sobre",
                                   min_value=1, value=5, step=1)

        st.subheader("🤝 Intercambio")
        n_people = st.number_input("7. Personas con las que intercambio",
                                   min_value=0, value=10, step=1)
        exchanged_already = st.number_input(
            "8. Figuritas ya conseguidas por intercambio",
            min_value=0, value=40, step=1)

        st.subheader("🎲 Simulación")
        n_sims = st.slider("9. Nº de simulaciones Monte Carlo",
                           min_value=100, max_value=5000, value=1000, step=100)
        budget = st.number_input("Presupuesto disponible (S/)",
                                 min_value=0.0, value=500.0, step=10.0)

        run = st.button("🚀 Ejecutar simulación", use_container_width=True,
                        type="primary")

    # Validación suave de consistencia entre 'tengo' y 'faltan'.
    if have + missing != n_total:
        st.info(f"ℹ️ Nota: *tengo* ({have}) + *faltan* ({missing}) = "
                f"{have + missing}, distinto del total ({n_total}). "
                f"Para los cálculos se usa **faltan = {missing}**.")

    missing = int(min(missing, n_total))

    # ---------------------- Métricas inmediatas ----------------------
    pct_complete = (n_total - missing) / n_total * 100 if n_total else 0
    # P(nueva en el siguiente sobre) = 1 - (poseídas/total)^per_pack
    owned_frac = (n_total - missing) / n_total if n_total else 1.0
    p_new_pack = 1 - owned_frac ** per_pack

    # ---------------------- Ejecución ----------------------
    if run:
        with st.spinner("Simulando escenarios (Monte Carlo)…"):
            sim_packs = simulate_panini_completion(
                int(n_total), int(missing), int(per_pack), int(n_sims))
            exch = estimate_exchange_savings(
                int(n_total), int(missing), int(duplicates),
                int(n_people), int(exchanged_already),
                int(per_pack), float(price))
            prob_budget = calculate_probability_completion(
                sim_packs, float(budget), float(price))

        # ------------------ Tarjetas KPI (fila 1) ------------------
        st.markdown("### 📊 Indicadores principales")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Álbum completado", f"{pct_complete:.1f}%",
                     f"{n_total - missing} de {n_total} figuritas")
        with c2:
            kpi_card("P(nueva en próximo sobre)", f"{p_new_pack*100:.1f}%",
                     f"{per_pack} figuritas por sobre")
        with c3:
            kpi_card("Sobres sin intercambio", f"{exch['packs_solo']:,.0f}",
                     "estimación Coupon Collector")
        with c4:
            kpi_card("Gasto sin intercambio", f"S/ {exch['cost_solo']:,.0f}",
                     "comprando todo en solitario")

        # ------------------ Tarjetas KPI (fila 2) ------------------
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            kpi_card("Sobres con intercambio",
                     f"{exch['packs_exchange']:,.0f}",
                     f"con {n_people} personas")
        with c6:
            kpi_card("Gasto con intercambio",
                     f"S/ {exch['cost_exchange']:,.0f}",
                     "comprando + canjeando")
        with c7:
            kpi_card("Ahorro por intercambio",
                     f"S/ {exch['savings']:,.0f}",
                     f"{exch['savings_pct']:.0f}% de ahorro")
        with c8:
            kpi_card("P(completar con presupuesto)",
                     f"{prob_budget*100:.1f}%",
                     f"con S/ {budget:,.0f}")

        # ------------------ Mensajes interpretativos ------------------
        st.markdown("### 🧭 Recomendaciones")
        message_box(f"💰 Tu ahorro estimado por intercambio es de "
                    f"<b>S/ {exch['savings']:,.0f}</b> "
                    f"({exch['savings_pct']:.0f}%).", "green")

        # Lógica de recomendación basada en el progreso y el ahorro.
        if pct_complete < 80:
            message_box("🛒 Comprar sobres todavía es conveniente: te faltan "
                        "muchas figuritas y la probabilidad de obtener nuevas "
                        "sigue siendo alta.", "blue")
        elif missing <= max(per_pack * 4, 20):
            message_box("🎯 Conviene comprar las figuritas faltantes "
                        "directamente: te quedan muy pocas y comprar sobres "
                        "al azar se vuelve caro (efecto cola del Coupon "
                        "Collector).", "amber")
        else:
            message_box("🤝 Conviene priorizar intercambios: ya tienes buena "
                        "parte del álbum y muchas repetidas para canjear.",
                        "green")

        if exch["savings_pct"] >= 40 and n_people > 0:
            message_box("🔁 El intercambio reduce tu gasto de forma muy "
                        "significativa. ¡Mantén tu red de coleccionistas!",
                        "green")

        # ------------------ Gráficos ------------------
        st.markdown("### 📈 Visualizaciones")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(plot_cost_curve(int(n_total), int(per_pack),
                                            float(price)),
                            use_container_width=True)
        with g2:
            st.plotly_chart(plot_cost_comparison(exch["cost_solo"],
                                                 exch["cost_exchange"]),
                            use_container_width=True)

        g3, g4 = st.columns(2)
        with g3:
            affordable = budget / price if price > 0 else None
            st.plotly_chart(plot_monte_carlo(sim_packs, affordable),
                            use_container_width=True)
        with g4:
            st.plotly_chart(
                plot_savings_curve(int(n_total), int(missing), int(duplicates),
                                   int(exchanged_already), int(per_pack),
                                   float(price)),
                use_container_width=True)

    else:
        # ------------------ Vista previa (antes de simular) ------------------
        st.markdown("### 📊 Vista rápida")
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Álbum completado", f"{pct_complete:.1f}%",
                     f"{n_total - missing} de {n_total} figuritas")
        with c2:
            kpi_card("P(nueva en próximo sobre)", f"{p_new_pack*100:.1f}%",
                     f"{per_pack} figuritas por sobre")
        with c3:
            kpi_card("Figuritas faltantes", f"{missing:,}",
                     "ajusta los parámetros y simula")
        st.info("👈 Ajusta los parámetros en la barra lateral y pulsa "
                "**🚀 Ejecutar simulación** para ver gasto, probabilidad, "
                "ahorro y gráficos interactivos.")

    # ---------------------- Explicación del modelo ----------------------
    with st.expander("📚 ¿Cómo funciona el modelo? (base matemática)"):
        st.markdown(
            r"""
**1. Coupon Collector Problem (estimación base).**
Para conseguir $m$ figuritas nuevas distintas de un álbum de $n$, el número
esperado de figuritas a comprar es:

$$ E[\text{figuritas}] = n \cdot H_m = n\left(1 + \tfrac12 + \dots + \tfrac1m\right) $$

Dividiendo entre las figuritas por sobre se obtienen los **sobres esperados**.
Para $n=638$ esto da ≈ **898 sobres** para completar desde cero, coherente con
el estudio.

**2. Simulación Monte Carlo (escenarios posibles).**
Se simula muchas veces la compra de sobres (cada figurita uniforme y con
reemplazo) hasta completar el álbum, y se registra cuántos sobres hicieron
falta. Esto entrega una **distribución** (no solo un promedio) y permite estimar
la **probabilidad de completar** dado un presupuesto.

**3. Modelo de intercambio (double dixie cup).**
Con $g$ coleccionistas que intercambian, el esfuerzo por persona se reduce según
$$ r(g) = \frac{\ln n + (g-1)\ln\ln n + \gamma}{g\,(\ln n + \gamma)} $$
Además, las **repetidas** y las **figuritas ya obtenidas por intercambio** se
canjean por faltantes, reduciendo aún más los sobres a comprar. Para $g=10$ el
gasto cae a ≈ **1/3**, en línea con los ≈ **321 sobres** del estudio.
            """
        )

    # ---------------------- Referencias ----------------------
    with st.expander("🔗 Referencias"):
        st.markdown(
            """
- **Rodrigo Gonzalez & Carlos A. Catania** — *The Panini collector problem*.
  https://rpubs.com/rodralez/panini
- Newman, D. J. & Shepp, L. (1960). *The double dixie cup problem*.
  American Mathematical Monthly. (Base del modelo con varios coleccionistas.)
- Ferrante, M. & Saltalamacchia, M. (2014). *The Coupon Collector's Problem*.
  Materials Matemàtics.
            """
        )

    st.caption("Hecho con Streamlit · Modelo educativo: las cifras son "
               "estimaciones probabilísticas, no garantías de gasto real.")


if __name__ == "__main__":
    main()
