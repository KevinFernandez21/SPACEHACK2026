# ManglarGYE — Documentación del Proyecto

## Visión General

**ManglarGYE** es una plataforma de análisis geoespacial y chatbot de inteligencia artificial para evaluar el rol protector de los manglares frente a inundaciones en el **Gran Guayaquil, Ecuador**. Combina datos satelitales en vivo desde Google Earth Engine (GEE), modelos de inundación y un asistente conversacional impulsado por Claude AI.

---

## Contexto del Problema

### Área de estudio: Gran Guayaquil

| Indicador | Dato |
|-----------|------|
| Población total | ~3.3 millones de habitantes |
| Guayaquil | 2.7M | Durán | 315K | Samborondón | 102K | Daule | 160K |
| Asentamientos informales en zonas inundables | ~30% de la población urbana |
| Concentración económica | ~25% del PIB de Ecuador |

### Vulnerabilidad climática

- Lluvias extremas: >70mm en un solo día (registros 2023–2025)
- Mareas del río Guayas: hasta 5 metros
- Fenómeno El Niño amplifica la frecuencia de eventos extremos
- Proyección IPCC AR6: aumento del nivel del mar de 0.3–0.6m para 2050
- Pérdidas por inundación 2023: >$50M USD (infraestructura)
- Costo promedio de daño urbano: $50,000 USD/ha

### El manglar como infraestructura natural

El Golfo de Guayaquil alberga el mayor estuario del Pacífico sudamericano. Sus manglares proveen:
- **Protección costera** frente a inundaciones
- **Captura de carbono**
- **Hábitat pesquero**

Sin embargo, entre 1970–2000 se perdió ~50% del manglar original, principalmente por la expansión camaronera.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────┐
│              Google Earth Engine                 │
│  - Imágenes Sentinel-2, SRTM DEM                │
│  - SERVIR-Amazonia (manglar 2018–2022)           │
│  - WorldPop (población), GHSL (urbano)           │
│  - Clasificación RF propia (2023–2025)           │
└──────────────────┬──────────────────────────────┘
                   │ API GEE
┌──────────────────▼──────────────────────────────┐
│           k4.ipynb — Análisis técnico            │
│  - Carga y procesamiento de manglares            │
│  - Modelo de inundación por escenarios           │
│  - Escenarios de restauración                    │
└──────────────────┬──────────────────────────────┘
                   │ Resultados
┌──────────────────▼──────────────────────────────┐
│        K5.ipynb — App Streamlit (Colab)          │
│  - UI con métricas en sidebar                    │
│  - Chatbot Claude AI (claude-sonnet-4)           │
│  - Contexto construido desde datos GEE           │
│  - Túnel público vía ngrok                       │
└─────────────────────────────────────────────────┘
```

---

## Notebook 1: `k4.ipynb` — Análisis Técnico

### Parte 1 — Contexto socioeconómico y político

Imprime un perfil completo del Gran Guayaquil con:
- Datos demográficos y distribución poblacional
- Vulnerabilidad climática e impacto económico de inundaciones
- Ecosistema de manglar (especies, servicios, pérdidas históricas)
- Stakeholders: GAD Municipal, MAATE, SENAGUA, SGR, INOCAR, sector camaronero, comunidades ancestrales
- Marco legal: Código Orgánico del Ambiente, Acuerdo Ministerial 129, Convenio de Ramsar, Acuerdo de París

---

### Parte 2 — Cobertura histórica de manglar

**Fuente:** SERVIR-Amazonia + Clasificación Random Forest propia

| Año | Fuente |
|-----|--------|
| 2018 | SERVIR (Asset GEE) |
| 2020 | SERVIR (Asset GEE) |
| 2022 | SERVIR (Asset GEE) |
| 2023 | Random Forest propio |
| 2024 | Random Forest propio |
| 2025 | Random Forest propio |

Los rásteres de manglar se cargan como `FeatureCollection`, se convierten a imagen y se recortan al AOI (Área de Interés) del Gran Guayaquil.

---

### Parte 3 — Modelo de inundación

#### Capas de datos utilizadas

| Capa | Fuente GEE |
|------|-----------|
| Elevación (DEM) | `USGS/SRTMGL1_003` |
| Población | `WorldPop/GP/100m/pop/ECU` (2020) |
| Superficie urbana | `JRC/GHSL/P2023A/GHS_BUILT_S/2020` |
| Manglar actual | SERVIR 2022 |

#### Modelo de atenuación del manglar

El manglar reduce el nivel efectivo de inundación en función de su ancho:

```
ancho_manglar = fastDistanceTransform → metros
atenuación = (ancho / 100m) × 30 cm  →  máximo 1.5m
nivel_efectivo = nivel_marea − atenuación
```

**Parámetros:**
- Atenuación: 30 cm por cada 100m de manglar
- Atenuación máxima: 1.5 m

#### Escenarios de inundación evaluados

| Escenario | Nivel (m) |
|-----------|-----------|
| Marea normal | 3.0 m |
| Marea alta | 4.0 m |
| Extremo: marea + lluvia | 5.0 m |
| Catastrófico | 6.0 m |

Para cada escenario se calcula:
- **Área inundada** con y sin manglar (ha)
- **Área protegida** por el manglar (ha)
- **Población expuesta** con y sin manglar
- **Área urbana protegida** (ha)
- **Daños evitados** (USD) = área urbana protegida × $50,000/ha

---

### Parte 4 — Escenarios de restauración

Se identifica el manglar perdido entre 2018 y 2022 que podría restaurarse:

```
manglar_perdido = manglar_2018 AND NOT manglar_2022
zona_restaurable = manglar_perdido AND NOT urbano AND elevación < 10m
```

Luego se recalcula el modelo de inundación extrema (5m) con el manglar restaurado para cuantificar la mejora en área inundada y población expuesta respecto al estado actual.

---

## Notebook 2: `K5.ipynb` — Aplicación Web (Streamlit en Google Colab)

### Propósito

Despliega una aplicación web interactiva con un chatbot de IA que responde preguntas sobre los manglares y las inundaciones del Gran Guayaquil, usando datos en vivo desde GEE.

### Flujo de ejecución

```
1. Instalar dependencias (streamlit, anthropic, pyngrok)
2. Escribir app_manglar.py con %%writefile
3. Iniciar Streamlit en puerto 8501 (background)
4. Crear túnel público con ngrok
5. Acceder a la app desde la URL ngrok generada
```

### Aplicación `app_manglar.py`

#### Carga de datos GEE (en tiempo real)

La función `load_all_data()` obtiene desde GEE:
- Áreas de manglar por año (2018–2025)
- Tasas de cambio entre períodos
- Resultados de los 4 escenarios de inundación
- Área restaurable potencial
- Métricas de protección poblacional y económica

#### Construcción del contexto para el LLM

La función `build_context()` ensambla todos los datos GEE en un texto estructurado que se entrega al modelo Claude como contexto del sistema, permitiendo respuestas basadas en datos reales y actualizados.

#### Interfaz Streamlit

**Sidebar:**
- Métricas clave: área de manglar actual, tendencia de cambio, población protegida
- Preguntas sugeridas para el usuario

**Chat principal:**
- Historial de conversación persistente en sesión
- Cada mensaje del usuario se envía a Claude con el contexto GEE completo
- Modelo utilizado: `claude-sonnet-4-20250514`

#### Stack tecnológico

| Componente | Tecnología |
|------------|-----------|
| Análisis geoespacial | Google Earth Engine Python API |
| UI web | Streamlit |
| LLM | Claude (Anthropic API) |
| Entorno de ejecución | Google Colab |
| Exposición pública | ngrok |
| Visualización estática | Matplotlib, Pandas |

---

## Flujo de datos completo

```
Satélites (Sentinel-2, SRTM)
        │
        ▼
Google Earth Engine Assets
  ├── MAN_2018, MAN_2020, MAN_2022  (SERVIR)
  ├── MANGROVE_2023/2024/2025       (RF propio)
  └── DEM, WorldPop, GHSL
        │
        ▼
Procesamiento GEE (k4.ipynb / app_manglar.py)
  ├── Rasterización y cálculo de áreas
  ├── Modelo de atenuación por ancho de manglar
  ├── Escenarios de inundación (3m–6m)
  └── Análisis de restauración
        │
        ▼
Contexto estructurado → Claude AI
        │
        ▼
Respuestas conversacionales en Streamlit
```

---

## Parámetros clave del modelo

| Parámetro | Valor |
|-----------|-------|
| Resolución espacial análisis | 10–30 m |
| Resolución población | 100 m |
| Atenuación manglar | 30 cm / 100 m |
| Atenuación máxima | 1.5 m |
| Costo daño urbano | $50,000 USD/ha |
| Elevación máxima restaurable | < 10 m |
| Escenarios de inundación | 3m, 4m, 5m, 6m |

---

## Marco regulatorio relevante

- **Código Orgánico del Ambiente (2017):** Art. 99–103 sobre manglares
- **Acuerdo Ministerial 129:** Concesiones en zonas de manglar
- **Plan Nacional de Manejo de Manglares** (MAATE)
- **PUGS Guayaquil 2022**, Plan de Adaptación al Cambio Climático 2023
- **ODS 11** (ciudades sostenibles), **ODS 13** (acción climática), **ODS 14** (vida submarina)
- Convenio de **Ramsar**, **Acuerdo de París**, Convención de **Cartagena**

---

## Actores clave (Stakeholders)

| Actor | Rol |
|-------|-----|
| GAD Municipal de Guayaquil | Planificación urbana y drenaje |
| MAATE | Custodio legal de los manglares |
| SENAGUA | Gestión hídrica y riesgo de inundación |
| SGR | Alertas tempranas |
| Comunidades ancestrales | Custodias de manglar |
| Sector camaronero | Principal driver histórico de deforestación |
| INOCAR | Datos de mareas y nivel del mar |

---

## Resumen ejecutivo

ManglarGYE demuestra cuantitativamente que los manglares del Gran Guayaquil actúan como infraestructura natural de protección ante inundaciones. El sistema permite:

1. **Monitorear** la evolución de la cobertura de manglar entre 2018 y 2025
2. **Modelar** el impacto de escenarios de inundación con y sin manglar
3. **Cuantificar** la población y el área urbana protegidas en términos económicos
4. **Identificar** zonas con potencial de restauración prioritaria
5. **Comunicar** estos resultados a tomadores de decisión mediante un chatbot conversacional con datos en vivo
