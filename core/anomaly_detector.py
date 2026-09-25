import numpy as np


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def detect_anomaly(fusion, sources=None, mission=None):
    """
    Detecta anomalias a partir dos dados já fundidos.

    O score combina sinais independentes:
    - inconsistência entre fontes;
    - baixa confiança da fusão;
    - frenagem brusca;
    - dispersão de velocidade;
    - aceleração extrema.

    É um detector heurístico de demonstração, não um modelo estatístico
    treinado com dados rotulados.
    """
    objects = fusion.get("objects")

    if objects is None or len(objects) == 0:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "signals": {},
            "explanation": "Não há objetos suficientes para avaliar anomalias.",
        }

    n = max(1, len(objects))

    # 1. Inconsistência entre fontes
    inconsistent = 0
    if "inconsistent" in objects.columns:
        inconsistent = int(
            objects["inconsistent"]
            .astype(str)
            .str.lower()
            .isin(["sim", "true", "1", "yes"])
            .sum()
        )
    inconsistency_signal = min(1.0, inconsistent / max(1.0, n * 0.10))

    # 2. Baixa confiança da fusão
    if "confidence" in objects.columns:
        confidence = objects["confidence"].astype(float).clip(0, 1)
        low_confidence = float((confidence < 0.70).mean())
        mean_confidence = float(confidence.mean())
    else:
        low_confidence = 0.0
        mean_confidence = _safe_float(fusion.get("mean_confidence", 1.0), 1.0)

    confidence_signal = min(
        1.0,
        0.60 * low_confidence + 0.40 * max(0.0, 0.80 - mean_confidence) / 0.40,
    )

    # 3. Frenagens bruscas
    if "acceleration_m_s2" in objects.columns:
        acceleration = objects["acceleration_m_s2"].astype(float)
        hard_braking = int((acceleration <= -3.0).sum())
        hard_braking_ratio = hard_braking / n
        braking_signal = min(1.0, hard_braking_ratio / 0.15)
        acceleration_signal = min(
            1.0,
            max(0.0, abs(float(acceleration.min())) - 2.0) / 3.0,
        )
    else:
        hard_braking = 0
        braking_signal = 0.0
        acceleration_signal = 0.0

    # 4. Dispersão anormal de velocidade
    if "speed_kmh" in objects.columns:
        speed = objects["speed_kmh"].astype(float)
        speed_std = _safe_float(speed.std(ddof=0))
        speed_signal = min(1.0, max(0.0, speed_std - 8.0) / 12.0)
    else:
        speed_std = 0.0
        speed_signal = 0.0

    # 5. Sinais específicos da missão.
    # Esses sinais não existem nos objetos V2X/RSU/Visão e, por isso,
    # precisam ser avaliados diretamente nas fontes adicionais do cenário.
    mission_signal = 0.0
    mission_reasons = []

    sources = sources or {}

    def source_df(*names):
        for name in names:
            value = sources.get(name)
            if value is not None and hasattr(value, "columns"):
                return value
        return None

    if mission in ("Agro Inteligente", "agro"):
        df = source_df("Sensores agrícolas", "Agro")
        if df is not None and len(df):
            temp = float(df["temperature_c"].max()) if "temperature_c" in df.columns else 0.0
            fuel = float(df["fuel_l_h"].max()) if "fuel_l_h" in df.columns else 0.0
            temp_signal = min(1.0, max(0.0, temp - 85.0) / 30.0)
            fuel_signal = min(1.0, max(0.0, fuel - 18.0) / 8.0)
            mission_signal = max(temp_signal, fuel_signal)
            if temp_signal > 0.05:
                mission_reasons.append(f"temperatura agrícola elevada ({temp:.1f} °C)")
            if fuel_signal > 0.05:
                mission_reasons.append(f"consumo elevado ({fuel:.1f} L/h)")

    elif mission in ("Economia Circular", "circular"):
        df = source_df("BMS", "Bateria")
        if df is not None and len(df):
            temp = float(df["temperature_c"].max()) if "temperature_c" in df.columns else 0.0
            resistance = float(df["internal_resistance_mohm"].max()) if "internal_resistance_mohm" in df.columns else 0.0
            cycles = float(df["cycles"].max()) if "cycles" in df.columns else 0.0
            temp_signal = min(1.0, max(0.0, temp - 40.0) / 15.0)
            resistance_signal = min(1.0, max(0.0, resistance - 20.0) / 12.0)
            cycle_signal = min(1.0, max(0.0, cycles - 1000.0) / 700.0)
            mission_signal = max(temp_signal, resistance_signal, cycle_signal)
            if temp_signal > 0.05:
                mission_reasons.append(f"temperatura da bateria elevada ({temp:.1f} °C)")
            if resistance_signal > 0.05:
                mission_reasons.append(f"resistência interna elevada ({resistance:.1f} mΩ)")
            if cycle_signal > 0.05:
                mission_reasons.append(f"ciclos elevados ({cycles:.0f})")

    elif mission in ("Lean Farm-to-Market", "farm_to_market"):
        df = source_df("Logística", "Farm-to-Market")
        if df is not None and len(df):
            traffic = float(df["traffic_index"].max()) if "traffic_index" in df.columns else 0.0
            travel = float(df["travel_time_min"].max()) if "travel_time_min" in df.columns else 0.0
            delay = float(df["loading_delay_min"].max()) if "loading_delay_min" in df.columns else 0.0
            traffic_signal = min(1.0, max(0.0, traffic - 0.70) / 0.30)
            travel_signal = min(1.0, max(0.0, travel - 180.0) / 110.0)
            delay_signal = min(1.0, max(0.0, delay - 15.0) / 30.0)
            mission_signal = max(traffic_signal, travel_signal, delay_signal)
            if traffic_signal > 0.05:
                mission_reasons.append(f"tráfego elevado ({traffic:.2f})")
            if travel_signal > 0.05:
                mission_reasons.append(f"tempo de viagem elevado ({travel:.0f} min)")
            if delay_signal > 0.05:
                mission_reasons.append(f"atraso de carregamento ({delay:.0f} min)")

    elif mission in ("Mobilidade para Pessoas", "mobility"):
        df = source_df("Visão urbana", "Mobilidade")
        if df is not None and len(df):
            distance = float(df["distance_m"].min()) if "distance_m" in df.columns else 999.0
            crossing = float(df["crossing_active"].sum()) if "crossing_active" in df.columns else 0.0
            distance_signal = min(1.0, max(0.0, 25.0 - distance) / 20.0)
            # Travessia isolada é um evento normal; ela só eleva o risco
            # quando ocorre junto de uma distância reduzida ao pedestre.
            crossing_signal = min(1.0, crossing / 2.0) * distance_signal
            mission_signal = max(distance_signal, crossing_signal)
            if distance_signal > 0.05:
                mission_reasons.append(f"distância reduzida a pedestre ({distance:.1f} m)")
            if crossing_signal > 0.05:
                mission_reasons.append(f"travessia ativa ({crossing:.0f})")

    score = (
        0.22 * inconsistency_signal
        + 0.16 * confidence_signal
        + 0.14 * braking_signal
        + 0.10 * acceleration_signal
        + 0.06 * speed_signal
        + 0.32 * mission_signal
    )

    score = float(np.clip(score, 0.0, 1.0))

    # Threshold relativamente conservador para a demonstração.
    is_anomaly = score >= 0.20

    signals = {
        "inconsistent_objects": inconsistent,
        "low_confidence_ratio": round(low_confidence, 3),
        "mean_confidence": round(mean_confidence, 3),
        "hard_braking_count": hard_braking,
        "speed_std_kmh": round(speed_std, 2),
        "inconsistency_signal": round(inconsistency_signal, 3),
        "confidence_signal": round(confidence_signal, 3),
        "braking_signal": round(braking_signal, 3),
        "acceleration_signal": round(acceleration_signal, 3),
        "speed_signal": round(speed_signal, 3),
    }

    active = []
    if inconsistency_signal > 0.05:
        active.append("inconsistência entre fontes")
    if confidence_signal > 0.05:
        active.append("baixa confiança da fusão")
    if braking_signal > 0.05:
        active.append("frenagem brusca")
    if acceleration_signal > 0.05:
        active.append("aceleração extrema")
    if speed_signal > 0.05:
        active.append("dispersão elevada de velocidade")
    active.extend(mission_reasons)

    explanation = (
        "Sinais observados: " + ", ".join(active) + "."
        if active
        else "Nenhum sinal relevante de anomalia foi identificado."
    )

    return {
        "is_anomaly": is_anomaly,
        "score": round(score, 3),
        "signals": signals,
        "explanation": explanation,
    }
