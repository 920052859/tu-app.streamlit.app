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
# 3b) FIGURITAS EXCLUSIVAS COCA-COLA  (canal de botellas, no de sobres)
# ===========================================================================
def estimate_cocacola_cost(cc_total: int, cc_have: int,
                           stickers_per_bottle: float, bottle_price: float,
                           random_draw: bool = True):
    """
    Estima cuántas botellas y cuánto dinero cuesta conseguir las figuritas
    EXCLUSIVAS de Coca-Cola, que no salen en sobres sino dentro de botellas
    (ej. 1.5 L). Es un sub-problema de colección aparte del álbum principal.

    Dos escenarios:
      - random_draw=True  : cada botella trae una figurita Coca-Cola AL AZAR
        del conjunto de `cc_total`. Conseguir las que faltan es un nuevo
        Coupon Collector:  E[figuritas] = cc_total * H_(cc_faltan).
        Es el caso realista y más caro (caen repetidas).
      - random_draw=False : puedes elegir/canjear exactamente las que faltan,
        así que solo necesitas `cc_faltan / figuritas_por_botella` botellas.
        Es el escenario MÍNIMO (mejor caso).

    Parameters
    ----------
    cc_total            : int    Total de figuritas Coca-Cola (ej. 14).
    cc_have             : int    Cuántas Coca-Cola ya tengo.
    stickers_per_bottle : float  Figuritas Coca-Cola por botella (ej. 1).
    bottle_price        : float  Precio de la botella 1.5 L (ej. 7.5).
    random_draw         : bool   Si las figuritas Coca-Cola salen al azar.

    Returns
    -------
    dict con: cc_missing, bottles, cost
    """
    cc_missing = max(cc_total - cc_have, 0)
    if cc_missing <= 0 or cc_total <= 0:
        return {"cc_missing": 0, "bottles": 0.0, "cost": 0.0}

    spb = max(stickers_per_bottle, 0.0001)
    if random_draw:
        # Coupon Collector sobre el conjunto pequeño de figuritas Coca-Cola.
        expected_stickers = cc_total * harmonic_number(cc_missing)
        bottles = expected_stickers / spb
    else:
        # Caso ideal: consigo exactamente las que me faltan.
        bottles = cc_missing / spb

    return {"cc_missing": cc_missing, "bottles": bottles,
            "cost": bottles * bottle_price}


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
COLOR_ACCENT = "#0f766e"     # teal (cohesión con el tema)


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
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --ink:#0a1f33; --pitch:#15803d; --grass:#22c55e; --gold:#f59e0b;
  --paper:#f6f8f4; --line:#e3e8e0; --muted:#6b7a72;
}

/* Cursor de pelota en todo el sitio (texto editable conserva el caret) */
* { cursor: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNiIgaGVpZ2h0PSIzNiIgdmlld0JveD0iMCAwIDM2IDM2Ij48Y2lyY2xlIGN4PSIxOCIgY3k9IjE4IiByPSIxNiIgZmlsbD0iI2ZmZmZmZiIgc3Ryb2tlPSIjMGYxNzJhIiBzdHJva2Utd2lkdGg9IjIiLz48cG9seWdvbiBwb2ludHM9IjE4LDkgMjUsMTQuNSAyMiwyMyAxNCwyMyAxMSwxNC41IiBmaWxsPSIjMGYxNzJhIi8+PHBhdGggZD0iTTE4LDkgTDE4LDMgTTI1LDE0LjUgTDMxLDEyIE0yMiwyMyBMMjYsMjkgTTE0LDIzIEwxMCwyOSBNMTEsMTQuNSBMNSwxMiIgc3Ryb2tlPSIjMGYxNzJhIiBzdHJva2Utd2lkdGg9IjEuNiIgZmlsbD0ibm9uZSIvPjwvc3ZnPgo=') 18 18, auto; }
input, textarea, select, [contenteditable] { cursor: text !important; }

html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
.block-container { padding-top:1.1rem; max-width:1200px; }

/* ---- HERO ---- */
.hero{
  background:
    repeating-linear-gradient(115deg, rgba(255,255,255,.035) 0 22px, rgba(255,255,255,0) 22px 44px),
    radial-gradient(130% 170% at 0% 0%, #114063 0%, var(--ink) 62%);
  border:1px solid rgba(255,255,255,.06); border-radius:18px;
  padding:30px 34px; margin-bottom:20px;
  box-shadow:0 14px 34px rgba(10,31,51,.28);
}
.hero .eyebrow{ font-family:'Anton',sans-serif; letter-spacing:.16em;
  text-transform:uppercase; color:var(--grass); font-size:.8rem; }
.hero h1{ font-family:'Anton',sans-serif; color:#fff; font-weight:400;
  font-size:3rem; line-height:.96; margin:.2rem 0 .4rem; text-transform:uppercase; }
.hero h1 .ball{ color:var(--grass); }
.hero p{ color:#aebfc9; font-size:1rem; margin:0; max-width:64ch; }

/* ---- Encabezados de sección ---- */
h2,h3{ font-family:'Inter',sans-serif !important; color:var(--ink) !important;
  font-weight:700 !important; }
.block-container h3{ border-top:1px solid var(--line);
  padding-top:.9rem; margin-top:1.4rem !important; }

/* ---- Tarjetas KPI estilo casilla de álbum ---- */
.kpi-card{ background:#fff; border:1px solid var(--line);
  border-left:5px solid var(--grass); border-radius:12px;
  padding:.95rem 1.1rem; height:100%;
  box-shadow:0 1px 2px rgba(10,31,51,.05);
  transition:transform .12s ease, box-shadow .12s ease; }
.kpi-card:hover{ transform:translateY(-2px);
  box-shadow:0 10px 24px rgba(10,31,51,.12); }
.kpi-label{ color:var(--muted); font-size:.72rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.07em; }
.kpi-value{ font-family:'Anton',sans-serif; font-weight:400;
  font-size:1.9rem; color:var(--ink); line-height:1.05; margin-top:.25rem; }
.kpi-sub{ color:#9aa8a0; font-size:.76rem; margin-top:.15rem; }

/* ---- Mensajes ---- */
.msg-box{ border-radius:12px; padding:.85rem 1.05rem; margin:.45rem 0;
  font-weight:600; border:1px solid transparent; }
.msg-green{ background:#eaf7ef; color:#14532d; border-color:#bbe7cb; }
.msg-blue { background:#eef4fb; color:#15406b; border-color:#c3d8ef; }
.msg-amber{ background:#fdf4e3; color:#7c4a08; border-color:#f3d79a; }

/* ---- Botón Ejecutar ---- */
.stButton > button{ background:var(--pitch) !important; color:#fff !important;
  border:0 !important; border-radius:11px !important; font-weight:700 !important;
  padding:.6rem 1rem !important;
  box-shadow:0 6px 16px rgba(21,128,61,.28) !important; }
.stButton > button:hover{ background:#126b34 !important; }
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
    st.markdown(
        '''<div class="hero">
              <div class="eyebrow">Coupon Collector · Monte Carlo · Intercambio</div>
              <h1>Panini <span class="ball">Smart</span> Collector</h1>
              <p>Calcula cuánto cuesta, con qué probabilidad y por qué camino
                 conviene completar tu álbum: comprar sobres, canjear repetidas
                 o ir directo a las que faltan.</p>
           </div>''',
        unsafe_allow_html=True)

    # ---------------------- Sidebar (inputs) ----------------------
    with st.sidebar:
        st.header("⚙️ Parámetros")

        st.subheader("📒 Álbum")
        n_total = st.number_input("1. Total de figuritas del álbum",
                                  min_value=1, value=980, step=1)
        have = st.number_input("2. Figuritas que ya tengo",
                               min_value=0, value=0, step=1)
        missing = st.number_input("3. Figuritas que me faltan",
                                  min_value=0, value=int(n_total - 0),
                                  step=1)
        duplicates = st.number_input("4. Figuritas repetidas",
                                     min_value=0, value=0, step=1)
        album_price = st.number_input("Precio del álbum vacío (S/)",
                                      min_value=0.0, value=49.90, step=1.0,
                                      format="%.2f")

        st.subheader("💸 Compra (sobres)")
        price = st.number_input("5. Precio por sobre (S/)",
                                min_value=0.1, value=4.20, step=0.10,
                                format="%.2f")
        per_pack = st.number_input("6. Figuritas por sobre",
                                   min_value=1, value=7, step=1)

        st.subheader("🥤 Figuritas Coca-Cola")
        st.caption("Figuritas exclusivas que salen en botellas, no en sobres.")
        cc_total = st.number_input("Figuritas exclusivas Coca-Cola",
                                   min_value=0, value=14, step=1)
        cc_have = st.number_input("Coca-Cola que ya tengo",
                                  min_value=0, value=0, step=1)
        cc_per_bottle = st.number_input("Figuritas Coca-Cola por botella",
                                        min_value=0.0, value=1.0, step=1.0,
                                        format="%.1f")
        bottle_price = st.number_input("Precio botella 1.5 L (S/)",
                                       min_value=0.0, value=7.50, step=0.50,
                                       format="%.2f")
        cc_random = st.checkbox("Las Coca-Cola salen al azar (repiten)",
                                value=True)

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

        run = st.button("🚀 Ejecutar simulación", width='stretch',
                        type="primary")

    # Validación suave de consistencia entre 'tengo' y 'faltan'.
    if have + missing != n_total:
        st.info(f"ℹ️ Nota: *tengo* ({have}) + *faltan* ({missing}) = "
                f"{have + missing}, distinto del total ({n_total}). "
                f"Para los cálculos se usa **faltan = {missing}**.")

    missing = int(min(missing, n_total))

    # ---------------------- Métricas inmediatas ----------------------
    # Las figuritas Coca-Cola son exclusivas de botellas: se separan del
    # "pool de sobres" para no contar su costo dos veces.
    cc_missing_now = max(int(cc_total) - int(cc_have), 0)
    pack_pool = max(int(n_total) - int(cc_total), 1)        # salen en sobres
    pack_missing = int(min(max(missing - cc_missing_now, 0), pack_pool))

    pct_complete = (n_total - missing) / n_total * 100 if n_total else 0
    # P(nueva en el siguiente sobre): sobre el pool de figuritas de sobre.
    owned_frac = (pack_pool - pack_missing) / pack_pool if pack_pool else 1.0
    p_new_pack = 1 - owned_frac ** per_pack

    # ---------------------- Ejecución ----------------------
    if run:
        with st.spinner("Simulando escenarios (Monte Carlo)…"):
            sim_packs = simulate_panini_completion(
                int(pack_pool), int(pack_missing), int(per_pack), int(n_sims))
            exch = estimate_exchange_savings(
                int(pack_pool), int(pack_missing), int(duplicates),
                int(n_people), int(exchanged_already),
                int(per_pack), float(price))
            prob_budget = calculate_probability_completion(
                sim_packs, float(budget), float(price))
            # Costo del canal Coca-Cola (botellas) y costo total del proyecto.
            cc = estimate_cocacola_cost(
                int(cc_total), int(cc_have), float(cc_per_bottle),
                float(bottle_price), bool(cc_random))
            total_solo = exch["cost_solo"] + cc["cost"] + album_price
            total_exchange = exch["cost_exchange"] + cc["cost"] + album_price

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

        # ------------------ Tarjetas KPI (fila 3: Coca-Cola + total) ------------------
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            kpi_card("Coca-Cola faltantes", f"{cc['cc_missing']:,}",
                     f"de {cc_total} exclusivas")
        with c10:
            kpi_card("Botellas Coca-Cola", f"{cc['bottles']:,.0f}",
                     "al azar" if cc_random else "eligiendo")
        with c11:
            kpi_card("Gasto Coca-Cola", f"S/ {cc['cost']:,.0f}",
                     f"botella 1.5 L a S/ {bottle_price:.2f}")
        with c12:
            kpi_card("Costo TOTAL del proyecto",
                     f"S/ {total_exchange:,.0f}",
                     f"con intercambio (sin: S/ {total_solo:,.0f})")

        # ------------------ Mensajes interpretativos ------------------
        st.markdown("### 🧭 Recomendaciones")
        message_box(f"💰 Tu ahorro estimado por intercambio es de "
                    f"<b>S/ {exch['savings']:,.0f}</b> "
                    f"({exch['savings_pct']:.0f}%).", "green")

        # Mensaje específico del canal Coca-Cola.
        if cc["cc_missing"] > 0:
            if cc_random:
                message_box(
                    f"🥤 Para las {cc['cc_missing']} figuritas Coca-Cola que te "
                    f"faltan necesitarías ~{cc['bottles']:,.0f} botellas "
                    f"(≈ S/ {cc['cost']:,.0f}), porque salen al azar y se "
                    f"repiten. ¡Aquí intercambiar o comprar las puntuales "
                    f"ahorra muchísimo!", "amber")
            else:
                message_box(
                    f"🥤 Para completar las Coca-Cola necesitas ~"
                    f"{cc['bottles']:,.0f} botellas (≈ S/ {cc['cost']:,.0f}).",
                    "blue")

        # Lógica de recomendación basada en el progreso y el ahorro.
        if pct_complete < 80:
            message_box("🛒 Comprar sobres todavía es conveniente: te faltan "
                        "muchas figuritas y la probabilidad de obtener nuevas "
                        "sigue siendo alta.", "blue")
        elif pack_missing <= max(per_pack * 4, 20):
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
            st.plotly_chart(plot_cost_curve(int(pack_pool), int(per_pack),
                                            float(price)),
                            width='stretch')
        with g2:
            st.plotly_chart(plot_cost_comparison(exch["cost_solo"],
                                                 exch["cost_exchange"]),
                            width='stretch')

        g3, g4 = st.columns(2)
        with g3:
            affordable = budget / price if price > 0 else None
            st.plotly_chart(plot_monte_carlo(sim_packs, affordable),
                            width='stretch')
        with g4:
            st.plotly_chart(
                plot_savings_curve(int(pack_pool), int(pack_missing),
                                   int(duplicates),
                                   int(exchanged_already), int(per_pack),
                                   float(price)),
                width='stretch')

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
