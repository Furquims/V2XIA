import pandas as pd


def _normalize_mission(mission):
    """Converte nomes exibidos na interface para identificadores internos."""
    mapping = {
        "Agro Inteligente": "agro",
        "Economia Circular": "circular",
        "Lean Farm-to-Market": "farm_to_market",
        "Mobilidade para Pessoas": "mobility",
        "agro": "agro",
        "circular": "circular",
        "farm_to_market": "farm_to_market",
        "mobility": "mobility",
    }
    return mapping.get(mission, mission)


def _number(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _risk_probability(score):
    """
    Converte score em probabilidades com separação clara entre os níveis.
    É uma função demonstrativa, não um modelo estatístico treinado.
    """
    score = max(0.0, min(1.0, float(score)))

    if score < 0.30:
        label = "Baixo"
    elif score < 0.60:
        label = "Moderado"
    else:
        label = "Alto"

    # Distribuição triangular simples para que o gráfico acompanhe
    # claramente o score.
    low = max(0.0, 1.0 - score * 3.0)
    high = max(0.0, (score - 0.30) / 0.70)
    moderate = max(0.0, 1.0 - abs(score - 0.50) / 0.30)

    probabilities = {
        "Baixo": low,
        "Moderado": moderate,
        "Alto": high,
    }

    total = sum(probabilities.values())

    if total <= 0:
        probabilities = {
            "Baixo": 1.0 if label == "Baixo" else 0.0,
            "Moderado": 1.0 if label == "Moderado" else 0.0,
            "Alto": 1.0 if label == "Alto" else 0.0,
        }
    else:
        probabilities = {
            key: value / total
            for key, value in probabilities.items()
        }

    return label, probabilities


def predict_risk(features, mission):
    """
    Classificador demonstrativo orientado ao contexto da missão.

    O protótipo utiliza regras explicáveis para demonstrar a arquitetura.
    Em uma evolução, essas regras podem ser substituídas por um modelo
    treinado com dados históricos rotulados.
    """

    # ------------------------------------------------------------
    # Entrada
    # ------------------------------------------------------------

    if isinstance(features, pd.DataFrame):
        r = features.iloc[0].to_dict() if not features.empty else {}
    elif isinstance(features, dict):
        r = features.copy()
    else:
        r = {}

    mission_id = _normalize_mission(mission)

    # ------------------------------------------------------------
    # Features gerais
    # ------------------------------------------------------------

    vehicle_count = _number(r, "vehicle_count")
    mean_speed = _number(r, "mean_speed_kmh")
    stopped_count = _number(r, "stopped_count")
    stopped_pct = _number(
        r,
        "stopped_pct",
        stopped_count / max(vehicle_count, 1.0),
    )
    inconsistencies = _number(
        r,
        "inconsistencies",
        _number(r, "fusion_inconsistencies"),
    )
    mean_confidence = _number(
        r,
        "mean_confidence",
        _number(r, "mean_fusion_confidence", 1.0),
    )
    hard_braking = _number(r, "hard_braking_count")
    mean_acc = _number(r, "mean_acceleration_m_s2")
    min_acc = _number(r, "min_acceleration_m_s2")

    # ------------------------------------------------------------
    # Score geral
    # ------------------------------------------------------------

    # Estes sinais são deliberadamente moderados porque as missões
    # possuem sinais próprios que serão adicionados abaixo.
    stopped_signal = min(1.0, stopped_pct / 0.45)
    braking_signal = min(
        1.0,
        hard_braking / max(2.0, vehicle_count * 0.20),
    )
    inconsistency_signal = min(
        1.0,
        inconsistencies / max(2.0, vehicle_count * 0.10),
    )
    confidence_signal = max(0.0, min(1.0, (0.85 - mean_confidence) / 0.30))
    speed_signal = max(0.0, min(1.0, (32.0 - mean_speed) / 25.0))
    acceleration_signal = max(0.0, min(1.0, abs(min_acc) / 5.0))

    general_score = (
        0.18 * stopped_signal
        + 0.22 * braking_signal
        + 0.16 * inconsistency_signal
        + 0.10 * confidence_signal
        + 0.16 * speed_signal
        + 0.08 * acceleration_signal
    )

    # ------------------------------------------------------------
    # Score específico da missão
    # ------------------------------------------------------------

    mission_score = 0.0

    if mission_id == "agro":
        temperature = _number(r, "max_temperature_c")
        mean_temperature = _number(r, "mean_temperature_c")
        fuel = _number(r, "mean_fuel_l_h")
        mean_fuel = _number(r, "mean_fuel_l_h")

        # O simulador normal gira em torno de 72 °C e 14 L/h.
        # O crítico gera 90–98 °C e 18–21 L/h.
        temperature_signal = max(
            0.0,
            min(1.0, (max(temperature, mean_temperature) - 78.0) / 20.0),
        )
        fuel_signal = max(
            0.0,
            min(1.0, (max(fuel, mean_fuel) - 15.5) / 5.0),
        )

        mission_score = (
            0.65 * temperature_signal
            + 0.35 * fuel_signal
        )

    elif mission_id == "circular":
        temperature = _number(r, "max_battery_temperature_c")
        resistance = _number(r, "mean_internal_resistance_mohm")
        max_resistance = _number(r, "max_internal_resistance_mohm")
        high_resistance = _number(r, "high_resistance_count")
        cycles = _number(r, "max_cycles")

        temperature_signal = max(
            0.0,
            min(1.0, (temperature - 40.0) / 15.0),
        )
        resistance_signal = max(
            0.0,
            min(
                1.0,
                (max(resistance, max_resistance) - 21.0) / 10.0,
            ),
        )
        count_signal = min(1.0, high_resistance / 3.0)
        cycles_signal = max(
            0.0,
            min(1.0, (cycles - 1000.0) / 500.0),
        )

        mission_score = (
            0.40 * temperature_signal
            + 0.40 * resistance_signal
            + 0.10 * count_signal
            + 0.10 * cycles_signal
        )

    elif mission_id == "farm_to_market":
        traffic = _number(r, "max_traffic_index", _number(r, "mean_traffic_index"))
        travel_time = _number(r, "max_travel_time_min", _number(r, "mean_travel_time_min"))
        loading_delay = _number(r, "max_loading_delay_min", _number(r, "mean_loading_delay_min"))

        # O erro anterior estava aqui: >40 min fazia sentido para uma
        # duração curta, mas o simulador trabalha com ~180 min normais.
        # Agora usamos desvio relativo a uma referência de 180 min.
        traffic_signal = max(
            0.0,
            min(1.0, (traffic - 0.65) / 0.30),
        )
        travel_signal = max(
            0.0,
            min(1.0, (travel_time - 190.0) / 80.0),
        )
        delay_signal = max(
            0.0,
            min(1.0, (loading_delay - 18.0) / 22.0),
        )

        mission_score = (
            0.40 * traffic_signal
            + 0.40 * travel_signal
            + 0.20 * delay_signal
        )

    elif mission_id == "mobility":
        crossing_count = _number(r, "crossing_active_count")
        min_distance = _number(
            r,
            "min_detected_distance_m",
            999.0,
        )

        crossing_signal = min(1.0, crossing_count / 4.0)
        distance_signal = max(
            0.0,
            min(1.0, (25.0 - min_distance) / 20.0),
        )

        mission_score = (
            0.60 * crossing_signal
            + 0.40 * distance_signal
        )

    # ------------------------------------------------------------
    # Combinação
    # ------------------------------------------------------------

    # A missão pesa mais que os sinais gerais quando existe um evento
    # específico, mantendo a fusão de sensores como componente transversal.
    score = (
        0.35 * general_score
        + 0.65 * mission_score
    )

    # Se a missão específica não detectou evento, sinais gerais ainda
    # podem elevar o risco.
    score = max(
        score,
        general_score * 0.65,
    )

    # Eventos extremos recebem um reforço explícito e explicável.
    if hard_braking >= max(5.0, vehicle_count * 0.20):
        score += 0.12

    if stopped_pct >= 0.40:
        score += 0.10

    if inconsistencies >= max(3.0, vehicle_count * 0.10):
        score += 0.08

    score = max(0.0, min(1.0, score))

    label, probabilities = _risk_probability(score)

    return {
        "label": label,
        "score": float(score),
        "probabilities": probabilities,
        "mission": mission,
    }
