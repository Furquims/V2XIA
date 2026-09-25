import streamlit as st
import pandas as pd
import time

from core.simulator import generate_scenario
from core.sensor_fusion import fuse_observations
from core.feature_engineering import build_features
from core.predictor import predict_risk
from core.anomaly_detector import detect_anomaly
from core.decision_layer import build_decision
from missions.registry import MISSION_CONFIG


# ============================================================
# CONFIGURAÇÃO
# ============================================================


def run_guided_demo(mission_key, mission_config):
    """Executa uma demonstração guiada usando um cenário crítico coerente."""
    demo_scenarios = {
        "agro": "Agro — condição operacional anômala",
        "circular": "Economia Circular — degradação de componente",
        "farm_to_market": "Farm-to-Market — atraso logístico",
        "mobility": "Mobilidade — congestionamento e frenagem",
    }

    scenario_name = demo_scenarios.get(mission_key, "Cenário crítico")
    placeholder = st.empty()

    steps = [
        ("COLETA", "Recebendo dados distribuídos de veículos, infraestrutura e sensores..."),
        ("FUSÃO", "Associando observações e verificando consistência espacial, temporal e física..."),
        ("FEATURE ENGINEERING", "Transformando as observações em variáveis para análise..."),
        ("MACHINE LEARNING", "Estimando o nível de risco a partir do comportamento observado..."),
        ("PREDIÇÃO", "Calculando probabilidade e classificação do risco..."),
        ("DECISION LAYER", "Transformando a predição em uma ação contextualizada..."),
    ]

    st.info(f"**Demonstração:** {scenario_name}")

    # Gera um cenário crítico real do simulador, em vez de números arbitrários.
    result_demo = generate_scenario(mission_key, scenario_name)

    for i, (title, description) in enumerate(steps, start=1):
        with placeholder.container():
            st.markdown(f"### {i}. {title}")
            st.write(description)
            st.progress(i / len(steps))
        time.sleep(0.45)

    placeholder.empty()
    return result_demo



def build_explainability(features, prediction, mission):
    """Gera uma explicação simples e transparente dos fatores que elevaram/reduziram o risco."""
    if hasattr(features, "iloc"):
        row = features.iloc[0].to_dict()
    elif isinstance(features, dict):
        row = features
    else:
        row = {}

    def num(name, default=0.0):
        try:
            value = row.get(name, default)
            return float(value) if value is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    # Fatores normalizados para visualização. São indicadores explicativos do
    # protótipo, não importância estatística validada de um modelo treinado.
    factors = {
        "Frenagens bruscas": min(1.0, num("hard_braking_count") / 5.0),
        "Veículos parados": min(1.0, num("stopped_pct") / 0.50),
        "Inconsistência entre sensores": min(1.0, num("inconsistencies") / 3.0),
        "Variação de velocidade": min(1.0, num("std_speed_kmh") / 20.0),
        "Baixa confiança da fusão": 1.0 - max(0.0, min(1.0, num("mean_confidence", 1.0))),
        "Aceleração extrema": min(1.0, max(abs(num("min_acceleration")), abs(num("max_acceleration"))) / 5.0),
    }

    factors = {k: float(max(0.0, min(1.0, v))) for k, v in factors.items()}
    ordered = sorted(factors.items(), key=lambda item: item[1], reverse=True)

    # Só apresenta os fatores com alguma contribuição observável.
    visible = [(name, value) for name, value in ordered if value > 0.01]
    if not visible:
        visible = [("Comportamento dentro do padrão", 0.05)]

    top = visible[:5]

    return {
        "factors": top,
        "mission": mission,
        "score": float(prediction.get("score", 0.0)),
        "note": (
            "Os valores representam indicadores explicativos do protótipo. "
            "Eles não devem ser interpretados como feature importance estatisticamente "
            "validada de um modelo treinado com dados reais."
        ),
    }



def build_scenario_comparison(mission_key, mission_config):
    """Gera dois cenários coerentes para comparação: normal e crítico."""
    names = {
        "Agro Inteligente": ("Fluxo normal", "Condição operacional anormal"),
        "Economia Circular": ("Fluxo normal", "Degradação de componente"),
        "Lean Farm-to-Market": ("Fluxo normal", "Atraso logístico"),
        "Mobilidade para Pessoas": ("Fluxo normal", "Congestionamento"),
        "agro": ("Fluxo normal", "Condição operacional anormal"),
        "circular": ("Fluxo normal", "Degradação de componente"),
        "farm_to_market": ("Fluxo normal", "Atraso logístico"),
        "mobility": ("Fluxo normal", "Congestionamento"),
    }

    normal_name, critical_name = names.get(
        mission_key,
        ("Cenário normal", "Cenário crítico")
    )

    # generate_scenario recebe primeiro a missão e depois o cenário.
    normal = generate_scenario(mission_key, normal_name)
    critical = generate_scenario(mission_key, critical_name)

    return {
        "normal": normal,
        "critical": critical,
        "normal_name": normal_name,
        "critical_name": critical_name,
    }


def summarize_scenario(result, mission_key):
    """Processa um cenário pelo mesmo pipeline utilizado pelo dashboard."""
    sources = result["sources"]
    fusion = fuse_observations(sources)
    features = build_features(fusion, mission_key, sources)
    prediction = predict_risk(features, mission_key)
    anomaly = detect_anomaly(fusion, sources=sources, mission=mission_key)
    decision = build_decision(mission_key, mission_key, features, prediction, anomaly)

    if hasattr(features, "iloc"):
        row = features.iloc[0]
    else:
        row = features

    def val(name, default=0.0):
        try:
            return float(row.get(name, default))
        except (TypeError, ValueError, AttributeError):
            return float(default)

    return {
        "result": result,
        "fusion": fusion,
        "features": features,
        "prediction": prediction,
        "anomaly": anomaly,
        "decision": decision,
        "vehicle_count": val("vehicle_count"),
        "mean_speed_kmh": val("mean_speed_kmh"),
        "stopped_count": val("stopped_count"),
        "hard_braking_count": val("hard_braking_count"),
        "mean_confidence": val("mean_confidence", 1.0),
        "inconsistencies": val("inconsistencies"),
    }



def validate_all_missions():
    """Executa o pipeline completo nas quatro missões, comparando normal e crítico."""
    mission_keys = [
        "Agro Inteligente",
        "Economia Circular",
        "Lean Farm-to-Market",
        "Mobilidade para Pessoas",
    ]

    results = {}

    for mission_key in mission_keys:
        comparison_raw = build_scenario_comparison(
            mission_key,
            MISSION_CONFIG[mission_key],
        )

        normal = summarize_scenario(
            comparison_raw["normal"],
            mission_key,
        )

        critical = summarize_scenario(
            comparison_raw["critical"],
            mission_key,
        )

        results[mission_key] = {
            "normal": normal,
            "critical": critical,
            "normal_name": comparison_raw["normal_name"],
            "critical_name": comparison_raw["critical_name"],
        }

    return results


def validation_status(normal, critical):
    """Classifica a qualidade da diferenciação do cenário."""
    normal_score = float(normal["prediction"].get("score", 0))
    critical_score = float(critical["prediction"].get("score", 0))

    speed_delta = abs(
        float(critical["mean_speed_kmh"]) -
        float(normal["mean_speed_kmh"])
    )

    stopped_delta = (
        float(critical["stopped_count"]) -
        float(normal["stopped_count"])
    )

    braking_delta = (
        float(critical["hard_braking_count"]) -
        float(normal["hard_braking_count"])
    )

    confidence_delta = (
        float(normal["mean_confidence"]) -
        float(critical["mean_confidence"])
    )

    score_delta = critical_score - normal_score

    signals = sum(
        [
            score_delta > 0.05,
            speed_delta > 3,
            stopped_delta > 1,
            braking_delta > 1,
            confidence_delta > 0.03,
        ]
    )

    if signals >= 3:
        status = "Diferenciação clara"
    elif signals >= 2:
        status = "Diferenciação parcial"
    else:
        status = "Revisar cenário"

    return {
        "status": status,
        "score_delta": score_delta,
        "speed_delta": speed_delta,
        "stopped_delta": stopped_delta,
        "braking_delta": braking_delta,
        "confidence_delta": confidence_delta,
    }



def mission_specific_validation_decision(mission_key, scenario_type, data):
    """Gera uma decisão contextual para a validação normal × crítica."""
    features = data.get("features")
    prediction = data.get("prediction", {})

    if hasattr(features, "iloc"):
        row = features.iloc[0].to_dict()
    elif isinstance(features, dict):
        row = features
    else:
        row = {}

    def val(name, default=0.0):
        try:
            return float(row.get(name, default))
        except (TypeError, ValueError, AttributeError):
            return float(default)

    critical = scenario_type == "critical"

    if mission_key == "Agro Inteligente":
        if critical:
            temperature = val("temperature_c", 0)
            fuel = val("fuel_l_h", 0)
            return {
                "title": "Inspecionar equipamento e avaliar intervenção preventiva",
                "action": (
                    "Priorizar a inspeção do equipamento que apresenta alteração "
                    "operacional e avaliar redução de carga ou parada preventiva."
                ),
                "reason": (
                    f"A decisão foi associada aos sinais de operação anômala, "
                    f"incluindo temperatura de referência de {temperature:.1f} °C "
                    f"e consumo de {fuel:.1f} L/h."
                ),
                "impact": "Reduzir risco operacional e preservar produtividade e eficiência.",
            }

        return {
            "title": "Manter operação e acompanhamento",
            "action": (
                "Manter a operação atual e continuar acompanhando temperatura, "
                "consumo e demais indicadores do equipamento."
            ),
            "reason": "Os indicadores permanecem próximos do comportamento operacional esperado.",
            "impact": "Manter produtividade sem intervenção desnecessária.",
        }

    if mission_key == "Economia Circular":
        if critical:
            temperature = val("temperature_c", 0)
            resistance = val("internal_resistance_mohm", 0)
            cycles = val("cycles", 0)
            return {
                "title": "Inspecionar componente e programar manutenção",
                "action": (
                    "Priorizar inspeção do componente com sinais de degradação e "
                    "programar manutenção antes de uma substituição prematura."
                ),
                "reason": (
                    f"Os indicadores apresentam temperatura de {temperature:.1f} °C, "
                    f"resistência interna de {resistance:.1f} mΩ e aproximadamente "
                    f"{cycles:.0f} ciclos."
                ),
                "impact": "Prolongar a vida útil do componente e reduzir descarte prematuro.",
            }

        return {
            "title": "Manter componente em operação",
            "action": (
                "Manter o componente em operação e continuar monitorando seus "
                "indicadores de degradação."
            ),
            "reason": "Os indicadores permanecem dentro do comportamento esperado para o cenário.",
            "impact": "Evitar manutenção ou substituição desnecessária.",
        }

    if mission_key == "Lean Farm-to-Market":
        if critical:
            travel = val("travel_time_min", 0)
            traffic = val("traffic_index", 0)
            delay = val("loading_delay_min", 0)
            return {
                "title": "Reavaliar rota e programação logística",
                "action": (
                    "Reavaliar a rota e a programação do lote afetado, considerando "
                    "o aumento do tempo de viagem, tráfego e atraso de carregamento."
                ),
                "reason": (
                    f"O cenário apresenta tempo de viagem de {travel:.0f} min, "
                    f"índice de tráfego de {traffic:.2f} e atraso de carregamento "
                    f"de {delay:.0f} min."
                ),
                "impact": "Reduzir variabilidade logística, atrasos e desperdícios na cadeia.",
            }

        return {
            "title": "Manter rota e programação",
            "action": (
                "Manter a rota e a programação logística atuais, continuando o "
                "monitoramento do fluxo."
            ),
            "reason": "Os indicadores logísticos permanecem próximos do padrão do cenário.",
            "impact": "Manter previsibilidade e estabilidade operacional.",
        }

    # Mobilidade para Pessoas
    if critical:
        speed = val("mean_speed_kmh", 0)
        stopped = val("stopped_count", 0)
        braking = val("hard_braking_count", 0)
        return {
            "title": "Priorizar investigação da região",
            "action": (
                "Priorizar a investigação da região e avaliar uma intervenção "
                "operacional diante do aumento de paradas e comportamentos anômalos."
            ),
            "reason": (
                f"O cenário apresenta velocidade média de {speed:.1f} km/h, "
                f"{stopped:.0f} veículos parados e {braking:.0f} ocorrências "
                f"de frenagem brusca."
            ),
            "impact": "Aumentar segurança, fluidez e previsibilidade da mobilidade.",
        }

    return {
        "title": "Manter monitoramento da região",
        "action": (
            "Manter o fluxo atual e continuar monitorando a região para identificar "
            "mudanças relevantes no comportamento da mobilidade."
        ),
        "reason": "Os indicadores permanecem compatíveis com o cenário operacional esperado.",
        "impact": "Preservar fluidez e segurança sem intervenção desnecessária.",
    }


st.set_page_config(
    page_title="V2X-AI",
    page_icon="◈",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    :root{
        --navy:#102A56;
        --indigo:#4F46E5;
        --bg:#F6F8FC;
        --line:#E4E9F2;
        --muted:#64748B;
    }

    .stApp{
        background:var(--bg);
    }

    .block-container{
        max-width:1280px;
        padding-top:1.5rem;
    }

    section[data-testid="stSidebar"]{
        background:#fff;
        border-right:1px solid var(--line);
    }

    .hero-title{
        color:var(--navy);
        font-size:2.6rem;
        font-weight:800;
        letter-spacing:-.04em;
        margin-bottom:0;
    }

    .hero-subtitle{
        color:var(--muted);
        font-size:1.05rem;
        margin-top:-.35rem;
        margin-bottom:1rem;
    }

    .section-label{
        color:var(--navy);
        font-size:1.35rem;
        font-weight:800;
    }

    .muted-text{
        color:var(--muted);
        font-size:.9rem;
    }

    .pipeline-node{
        display:inline-block;
        padding:.45rem .7rem;
        margin:.15rem;
        border-radius:9px;
        background:#F0F2FF;
        color:var(--navy);
        font-size:.84rem;
        font-weight:650;
    }

    .pipeline-arrow{
        color:var(--indigo);
        font-weight:800;
        margin:0 .1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "generated_mission" not in st.session_state:
    st.session_state.generated_mission = None

if "generated_scenario" not in st.session_state:
    st.session_state.generated_scenario = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## V2X-AI")
st.sidebar.caption("Interactive Prototype · V0.5")

mission = st.sidebar.radio(
    "Missão",
    list(MISSION_CONFIG),
    format_func=lambda x: MISSION_CONFIG[x]["short"],
)

cfg = MISSION_CONFIG[mission]

scenario = st.sidebar.selectbox(
    "Cenário",
    cfg["scenarios"],
)



st.sidebar.markdown("---")
st.sidebar.subheader("Demonstração")
st.sidebar.caption("Executa automaticamente um cenário crítico e percorre a arquitetura completa.")

if st.sidebar.button("▶ Executar demonstração", width="stretch"):
    demo_result = run_guided_demo(mission, MISSION_CONFIG[mission])
    st.session_state.result = demo_result
    st.session_state.generated = True
    st.session_state.demo_mode = False
if "comparison" not in st.session_state:
    st.session_state.comparison = None
if "validation" not in st.session_state:
    st.session_state.validation = None
    st.session_state.demo_mode = True
    st.rerun()

if st.sidebar.button(
    "▶ Gerar cenário",
    type="primary",
    width="stretch",
):
    try:
        with st.spinner(
            "Gerando cenário e processando os dados..."
        ):
            st.session_state.result = generate_scenario(
                mission,
                scenario,
            )

            st.session_state.generated_mission = mission
            st.session_state.generated_scenario = scenario

    except Exception as exc:
        st.session_state.result = None

        st.error("Erro ao gerar o cenário.")
        st.exception(exc)


# ============================================================
# CABEÇALHO
# ============================================================

st.markdown(
    '<div class="hero-title">V2X-AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">'
    'Transformando dados distribuídos em inteligência preditiva.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# PIPELINE
# ============================================================

pipeline_nodes = [
    "Fontes distribuídas",
    "Fusão",
    "Features",
    "Machine Learning",
    "Predição",
    "Decision Layer",
]

pipeline_text = " → ".join(pipeline_nodes)

st.info(pipeline_text)

st.caption(cfg["question"])


# ============================================================
# SEM RESULTADO
# ============================================================

if not st.session_state.result:
    st.info(
        "Escolha uma missão e um cenário no menu lateral "
        "e clique em Gerar cenário."
    )
    st.stop()


# ============================================================
# PROCESSAMENTO
# ============================================================

result = st.session_state.result

if st.session_state.get("demo_mode", False):
    st.success("Modo demonstração concluído. O resultado abaixo percorre a cadeia completa de Coleta → Fusão → Features → IA → Predição → Decisão.")
    if st.button("Voltar ao modo interativo", width="stretch"):
        st.session_state.demo_mode = False
        st.rerun()


sources = result["sources"]

fusion = fuse_observations(
    sources
)

features = build_features(
    fusion,
    mission,
    sources,
)

prediction = predict_risk(
    features,
    mission,
)

anomaly = detect_anomaly(
    fusion,
    sources=sources,
    mission=mission,
)

decision = build_decision(
    mission,
    scenario,
    features,
    prediction,
    anomaly,
)


# ============================================================
# KPIs
# ============================================================

k = st.columns(5)

k[0].metric(
    "Objetos",
    fusion["object_count"],
)

k[1].metric(
    "Fontes",
    len(sources),
)

k[2].metric(
    "Confiança",
    f'{fusion["mean_confidence"] * 100:.1f}%',
)

k[3].metric(
    "Inconsistências",
    fusion["inconsistencies"],
)

k[4].metric(
    "Predição",
    prediction["label"],
)


# ============================================================
# TABS
# ============================================================

t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    [
        "01 · Percepção",
        "02 · Fusão",
        "03 · Inteligência",
        "04 · Decisão",
        "05 · Missões",
        "06 · Comparação",
        "07 · Validação",
    ]
)


# ============================================================
# 01 — PERCEPÇÃO
# ============================================================

with t1:

    st.subheader("Fontes distribuídas")

    st.caption(
        "Cada fonte representa uma percepção independente do ambiente."
    )

    cols = st.columns(
        min(4, len(sources))
    )

    for i, (name, df) in enumerate(sources.items()):

        with cols[i % len(cols)]:

            confidence = 0.0

            if (
                "confidence" in df.columns
                and not df.empty
            ):
                confidence = (
                    df["confidence"].mean() * 100
                )

            with st.container(border=True):

                st.markdown(
                    f"**{name}**"
                )

                st.caption(
                    f"{len(df)} observações · "
                    f"confiança média {confidence:.1f}%"
                )

    st.markdown("### Inspecionar fonte")

    choice = st.selectbox(
        "Fonte",
        list(sources),
        label_visibility="collapsed",
    )

    st.dataframe(
        sources[choice],
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 02 — FUSÃO
# ============================================================

with t2:

    st.subheader("Fusão de percepção")

    st.caption(
        "Associação espaço-temporal + consistência + confiança "
        "→ percepção unificada."
    )

    q = fusion["quality"]

    c = st.columns(4)

    c[0].metric(
        "Qualidade espacial",
        f'{q["spatial"] * 100:.0f}%',
    )

    c[1].metric(
        "Qualidade temporal",
        f'{q["temporal"] * 100:.0f}%',
    )

    c[2].metric(
        "Consistência física",
        f'{q["physical"] * 100:.0f}%',
    )

    c[3].metric(
        "Qualidade final",
        f'{q["final"] * 100:.0f}%',
    )

    st.markdown("### Objetos fusionados")

    st.dataframe(
        fusion["objects"],
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Associação de observações")

    st.dataframe(
        fusion["association_sample"],
        width="stretch",
        hide_index=True,
    )

    if fusion["inconsistencies"]:

        st.warning(
            f'{fusion["inconsistencies"]} '
            "observação(ões) inconsistentes; "
            "a confiança foi penalizada antes da fusão."
        )

    else:

        st.success(
            "As fontes apresentaram comportamento "
            "consistente no cenário."
        )


# ============================================================
# 03 — INTELIGÊNCIA
# ============================================================

with t3:

    st.subheader("Feature Engineering + IA")

    st.dataframe(
        features,
        width="stretch",
        hide_index=True,
    )

    probs = prediction["probabilities"]

    c = st.columns(3)

    c[0].metric(
        "Baixo",
        f'{probs["Baixo"] * 100:.0f}%',
    )

    c[1].metric(
        "Moderado",
        f'{probs["Moderado"] * 100:.0f}%',
    )

    c[2].metric(
        "Alto",
        f'{probs["Alto"] * 100:.0f}%',
    )

    # --------------------------------------------------------
    # GRÁFICO
    # --------------------------------------------------------

    chart_df = pd.DataFrame(
        {
            "Risco": list(probs.keys()),
            "Probabilidade": [
                value * 100
                for value in probs.values()
            ],
        }
    )

    st.markdown("### Probabilidade de risco")

    st.bar_chart(
        chart_df,
        x="Risco",
        y="Probabilidade",
        height=320,
        width="stretch",
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    st.markdown("### Resultado da predição")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predição",
            prediction["label"],
        )

    with result_col2:
        st.metric(
            "Score demonstrativo",
            f'{prediction["score"]:.2f}',
        )

    # --------------------------------------------------------
    # ANOMALIA
    # --------------------------------------------------------

    if anomaly["is_anomaly"]:

        st.error(
            "Detecção de anomalia: DETECTADA "
            f'· score {anomaly["score"]:.2f}'
        )

    else:

        st.success(
            "Detecção de anomalia: não detectada "
            f'· score {anomaly["score"]:.2f}'
        )

    st.caption(anomaly.get("explanation", ""))

    signals = anomaly.get("signals", {})
    if signals:
        signal_cols = st.columns(3)
        signal_cols[0].metric(
            "Objetos inconsistentes",
            signals.get("inconsistent_objects", 0),
        )
        signal_cols[1].metric(
            "Frenagens bruscas",
            signals.get("hard_braking_count", 0),
        )
        signal_cols[2].metric(
            "Confiança média",
            f'{signals.get("mean_confidence", 0):.2f}',
        )


# ============================================================
# 04 — DECISION LAYER
# ============================================================

with t4:

    st.subheader("Decision Layer")

    st.caption(
        "A previsão é transformada em uma resposta "
        "específica para a missão."
    )

    st.markdown(
        f'### {decision["event"]}'
    )

    decision_col1, decision_col2 = st.columns(2)

    with decision_col1:

        st.markdown("**Prioridade**")

        if decision["priority"] == "Alta":

            st.error(
                decision["priority"]
            )

        elif decision["priority"] == "Média":

            st.warning(
                decision["priority"]
            )

        else:

            st.success(
                decision["priority"]
            )

    with decision_col2:

        st.markdown("**Risco identificado**")

        st.info(
            f'{prediction["label"]} · '
            f'score {prediction["score"]:.2f}'
        )

    st.markdown("**Ação sugerida**")
    st.write(decision["action"])

    st.markdown("**Justificativa**")
    st.write(decision["justification"])

    st.markdown("**Impacto esperado**")
    st.write(decision["impact"])


# ============================================================
# 05 — MISSÕES
# ============================================================

with t5:

    st.subheader(
        "Uma arquitetura. Quatro aplicações."
    )

    for key, item in MISSION_CONFIG.items():

        with st.container(
            border=(key == mission)
        ):

            st.markdown(
                f"### {item['short']}"
            )

            st.write(
                item["answer"]
            )



with t7:
    st.subheader("Validação das 4 missões")
    st.caption(
        "Executa o mesmo pipeline em cenário normal e crítico para cada missão "
        "e verifica se a arquitetura produz uma mudança observável."
    )

    if st.button("▶ Executar validação completa", width="stretch"):
        with st.spinner("Executando 8 cenários pelo pipeline completo..."):
            st.session_state.validation = validate_all_missions()

    validation = st.session_state.get("validation")

    if validation is None:
        st.info(
            "Execute a validação para verificar as quatro missões antes da publicação."
        )
    else:
        rows = []

        mission_labels = {
            "Agro Inteligente": "Agro Inteligente",
            "Economia Circular": "Economia Circular",
            "Lean Farm-to-Market": "Lean Farm-to-Market",
            "Mobilidade para Pessoas": "Mobilidade para Pessoas",
        }

        for mission_key, data in validation.items():
            normal = data["normal"]
            critical = data["critical"]
            check = validation_status(normal, critical)

            rows.append(
                {
                    "Missão": mission_labels.get(mission_key, mission_key),
                    "Risco normal": normal["prediction"]["label"],
                    "Risco crítico": critical["prediction"]["label"],
                    "Δ score": round(check["score_delta"], 2),
                    "Δ velocidade": f"{check['speed_delta']:.1f} km/h",
                    "Δ parados": f"{check['stopped_delta']:.0f}",
                    "Δ frenagens": f"{check['braking_delta']:.0f}",
                    "Δ confiança": f"{check['confidence_delta']:.1%}",
                    "Status": check["status"],
                }
            )

        validation_df = pd.DataFrame(rows)

        st.markdown("### Resultado geral")
        st.dataframe(
            validation_df,
            hide_index=True,
            width="stretch",
        )

        clear_count = sum(
            row["Status"] == "Diferenciação clara"
            for row in rows
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Missões avaliadas", len(rows))
        c2.metric("Diferenciação clara", clear_count)
        c3.metric(
            "Resultado",
            "OK" if clear_count == len(rows) else "Revisar",
        )

        st.markdown("### Análise por missão")

        for mission_key, data in validation.items():
            normal = data["normal"]
            critical = data["critical"]
            check = validation_status(normal, critical)

            with st.container(border=True):
                st.markdown(f"**{mission_labels.get(mission_key, mission_key)}**")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.write("**Normal**")
                    st.metric(
                        "Risco",
                        normal["prediction"]["label"],
                        f"{normal['prediction'].get('score', 0):.2f}",
                    )

                with c2:
                    st.write("**Crítico**")
                    st.metric(
                        "Risco",
                        critical["prediction"]["label"],
                        f"{critical['prediction'].get('score', 0):.2f}",
                    )

                with c3:
                    st.write("**Validação**")
                    if check["status"] == "Diferenciação clara":
                        st.success(check["status"])
                    elif check["status"] == "Diferenciação parcial":
                        st.warning(check["status"])
                    else:
                        st.error(check["status"])

                st.caption(
                    f"Velocidade: {normal['mean_speed_kmh']:.1f} → "
                    f"{critical['mean_speed_kmh']:.1f} km/h · "
                    f"Parados: {normal['stopped_count']:.0f} → "
                    f"{critical['stopped_count']:.0f} · "
                    f"Frenagens: {normal['hard_braking_count']:.0f} → "
                    f"{critical['hard_braking_count']:.0f}"
                )

                normal_decision = mission_specific_validation_decision(
                    mission_key,
                    "normal",
                    normal,
                )
                critical_decision = mission_specific_validation_decision(
                    mission_key,
                    "critical",
                    critical,
                )

                st.markdown("**Decisão normal**")
                st.info(
                    f"**{normal_decision['title']}**\n\n"
                    f"{normal_decision['action']}\n\n"
                    f"_{normal_decision['reason']}_"
                )

                st.markdown("**Decisão crítica**")
                st.warning(
                    f"**{critical_decision['title']}**\n\n"
                    f"{critical_decision['action']}\n\n"
                    f"_{critical_decision['reason']}_"
                )

                st.caption(
                    f"Impacto esperado — normal: {normal_decision['impact']} "
                    f"| crítico: {critical_decision['impact']}"
                )

        st.info(
            "Esta validação demonstra a coerência funcional do protótipo com dados "
            "sintéticos. Ela não substitui validação estatística com dados reais."
        )

# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "V0.5 · Dados sintéticos e IA demonstrativa. "
    "SUMO, PostgreSQL e hardware não são necessários "
    "para executar esta versão."
)

with t6:
    st.subheader("Comparação de cenários")
    st.caption(
        "O mesmo pipeline é executado em dois estados do ambiente para evidenciar "
        "como a mudança de comportamento altera a percepção, a predição e a decisão."
    )

    if st.button("↻ Gerar comparação normal × crítico", width="stretch"):
        with st.spinner("Executando os dois cenários pelo pipeline completo..."):
            comparison_raw = build_scenario_comparison(mission, MISSION_CONFIG[mission])
            comparison = {
                "normal": summarize_scenario(comparison_raw["normal"], mission),
                "critical": summarize_scenario(comparison_raw["critical"], mission),
                "normal_name": comparison_raw["normal_name"],
                "critical_name": comparison_raw["critical_name"],
            }
            st.session_state.comparison = comparison

    comparison = st.session_state.get("comparison")

    if comparison is None:
        st.info("Clique em **Gerar comparação normal × crítico** para executar os dois cenários.")
    else:
        normal = comparison["normal"]
        critical = comparison["critical"]

        st.markdown("### Visão geral")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Normal** — {comparison['normal_name']}")
            st.metric("Risco", normal["prediction"]["label"])
        with c2:
            st.markdown(f"**Crítico** — {comparison['critical_name']}")
            st.metric("Risco", critical["prediction"]["label"])

        rows = [
            ("Veículos", normal["vehicle_count"], critical["vehicle_count"], ""),
            ("Velocidade média", normal["mean_speed_kmh"], critical["mean_speed_kmh"], " km/h"),
            ("Veículos parados", normal["stopped_count"], critical["stopped_count"], ""),
            ("Frenagens bruscas", normal["hard_braking_count"], critical["hard_braking_count"], ""),
            ("Confiança da fusão", normal["mean_confidence"] * 100, critical["mean_confidence"] * 100, "%"),
            ("Inconsistências", normal["inconsistencies"], critical["inconsistencies"], ""),
        ]

        table = pd.DataFrame(
            {
                "Indicador": [r[0] for r in rows],
                "Normal": [
                    f"{r[1]:.1f}{r[3]}" if r[3] != "%" else f"{r[1]:.1f}%"
                    for r in rows
                ],
                "Crítico": [
                    f"{r[2]:.1f}{r[3]}" if r[3] != "%" else f"{r[2]:.1f}%"
                    for r in rows
                ],
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")

        st.markdown("### Variação observada")

        for label, normal_value, critical_value, suffix in rows:
            if abs(normal_value) < 1e-9:
                variation_text = "Novo indicador no cenário crítico"
            else:
                variation = (critical_value - normal_value) / abs(normal_value)
                variation_text = f"{variation:+.0%}"

            c1, c2, c3 = st.columns([2.2, 1, 2])
            c1.write(label)
            c2.write(f"{critical_value:.1f}{suffix}")
            c3.write(variation_text)

        st.markdown("---")
        st.markdown("### O que mudou na interpretação?")

        if mission == "agro":
            explanation = (
                "A comparação mostra como alterações nas variáveis operacionais podem "
                "mudar a classificação de risco e gerar uma recomendação voltada à "
                "inspeção da operação."
            )
        elif mission == "circular":
            explanation = (
                "A comparação evidencia como sinais associados ao uso severo ou "
                "degradação podem elevar o risco e direcionar a decisão para inspeção "
                "do componente antes de uma substituição prematura."
            )
        elif mission == "farm_to_market":
            explanation = (
                "A comparação evidencia como mudanças no fluxo e nos indicadores "
                "logísticos podem elevar o risco de atraso e direcionar a decisão para "
                "avaliação de rota, programação ou recebimento."
            )
        else:
            explanation = (
                "A comparação evidencia como alterações no comportamento do tráfego, "
                "paradas e frenagens podem elevar o risco e direcionar a decisão para "
                "investigação da área ou ajuste da operação."
            )

        st.info(explanation)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Decisão — Normal**")
            st.write(normal["decision"].get("action", "Sem ação definida."))
        with d2:
            st.markdown("**Decisão — Crítico**")
            st.write(critical["decision"].get("action", "Sem ação definida."))

        st.markdown("### Evidência da evolução")
        st.write(
            "O ponto central da comparação é que a mesma arquitetura de dados, fusão, "
            "predição e decisão permanece constante; o que muda é o estado observado "
            "do ambiente e, consequentemente, a resposta contextual."
        )

