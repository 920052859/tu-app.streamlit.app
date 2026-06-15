# ⚽ Panini Smart Collector

**Simulador de gasto, probabilidad e intercambio** para completar un álbum de
figuritas (cromos) tipo Panini, inspirado en el *Panini Collector Problem*.

App interactiva en **Python + Streamlit** que estima cuánto te costará
completar el álbum, con qué probabilidad lo lograrás según tu presupuesto, y
cuánto puedes ahorrar intercambiando figuritas con otros coleccionistas.

> Caso de referencia (álbum Qatar 2022): **638 figuritas**, **5 por sobre**.
> Sin intercambio ≈ **899 sobres**; con 10 coleccionistas ≈ **321 sobres**.

---

## 🎯 Objetivo

Ayudar a un coleccionista a tomar decisiones informadas:

- ¿Cuántos sobres y cuánto dinero necesito para completar el álbum?
- ¿Qué probabilidad tengo de completarlo con mi presupuesto?
- ¿Cuánto ahorro si intercambio repetidas en lugar de comprar todo?
- ¿Me conviene seguir comprando sobres, priorizar intercambios, o comprar
  directamente las figuritas que faltan?

---

## 🚀 Instalación y ejecución

```bash
pip install streamlit numpy pandas plotly
streamlit run app.py
```

La app se abre en el navegador (por defecto `http://localhost:8501`).

---

## 🕹️ Parámetros de entrada (barra lateral)

| # | Parámetro | Descripción |
|---|-----------|-------------|
| 1 | Total de figuritas | Tamaño del álbum (ej. 638) |
| 2 | Figuritas que ya tengo | Progreso actual |
| 3 | Figuritas que me faltan | Pendientes por conseguir |
| 4 | Figuritas repetidas | Inventario para intercambiar |
| 5 | Precio por sobre | En S/ |
| 6 | Figuritas por sobre | Normalmente 5 |
| 7 | Personas con las que intercambio | Tamaño de tu red |
| 8 | Figuritas ya obtenidas por intercambio | Ventaja acumulada |
| 9 | Nº de simulaciones Monte Carlo | Precisión del cálculo |
| – | Presupuesto disponible | Para la probabilidad de completar |

---

## 📊 Qué calcula

- Porcentaje del álbum completado.
- Probabilidad de obtener una figurita nueva en el siguiente sobre.
- Sobres y gasto estimados **sin** intercambio.
- Sobres y gasto estimados **con** intercambio.
- Ahorro estimado y porcentaje de ahorro.
- Probabilidad de completar el álbum según el presupuesto ingresado.
- **Canal Coca-Cola:** costo aparte de las figuritas exclusivas que salen en
  botellas (ej. 14 figuritas en botellas 1.5 L), modeladas como un Coupon
  Collector pequeño e independiente, más el **costo total del proyecto**
  (álbum + sobres + botellas).

Y muestra 4 gráficos interactivos (Plotly): curva de gasto, comparación
con/sin intercambio, distribución Monte Carlo y curva de ahorro.

---

## 🧮 Base matemática

**1. Coupon Collector Problem.** Conseguir `m` figuritas nuevas de un álbum de
`n` requiere, en promedio:

```
E[figuritas] = n · H_m = n · (1 + 1/2 + ... + 1/m)
```

Dividiendo entre las figuritas por sobre se obtienen los sobres esperados.
Para `n = 638` esto da ≈ 898 sobres para completar desde cero.

**2. Simulación Monte Carlo.** Se simula muchas veces la compra de sobres
(cada figurita uniforme y con reemplazo) hasta completar el álbum. Esto entrega
una **distribución** de sobres necesarios y permite estimar la probabilidad de
completar dado un presupuesto.

**3. Modelo de intercambio (*double dixie cup* / Newman–Shepp).** Con `g`
coleccionistas que intercambian, el esfuerzo por persona se reduce según:

```
r(g) = [ ln n + (g-1)·ln(ln n) + γ ] / [ g · (ln n + γ) ]
```

Además, las repetidas y las figuritas ya obtenidas por intercambio se canjean
por faltantes. Para `g = 10` el gasto cae a ≈ 1/3, en línea con el estudio.

---

## 🗂️ Estructura del código (`app.py`)

| Función | Rol |
|---------|-----|
| `coupon_collector_estimate()` | Estimación analítica base |
| `simulate_panini_completion()` | Simulación Monte Carlo (vectorizada) |
| `estimate_exchange_savings()` | Modelo de intercambio y ahorro |
| `estimate_cocacola_cost()` | Costo de las figuritas exclusivas de botellas |
| `calculate_probability_completion()` | Probabilidad según presupuesto |
| `main()` | Interfaz Streamlit (sidebar, KPIs, gráficos) |

---

## 🔗 Referencias

- Rodrigo Gonzalez & Carlos A. Catania — *The Panini collector problem*.
  <https://rpubs.com/rodralez/panini>
- Newman, D. J. & Shepp, L. (1960). *The double dixie cup problem*.
  *American Mathematical Monthly*.
- Ferrante, M. & Saltalamacchia, M. (2014). *The Coupon Collector's Problem*.
  *Materials Matemàtics*.

---

> ⚠️ Modelo educativo: las cifras son estimaciones probabilísticas, no
> garantías de gasto real.
