import pandas as pd


def build_features(fusion, mission, sources):
    """
    Constrói as features utilizadas pela camada de inteligência.

    Parâmetros
    ----------
    fusion : dict
        Resultado produzido por fuse_observations().

    mission : str
        Identificador da missão selecionada.

    sources : dict
        Fontes originais utilizadas na fusão.

    Retorno
    -------
    pd.DataFrame
        Tabela com as features utilizadas pelo modelo.
    """

    mission_map = {
        "Agro Inteligente": "agro",
        "Economia Circular": "circular",
        "Lean Farm-to-Market": "farm_to_market",
        "Mobilidade para Pessoas": "mobility",
    }
    mission = mission_map.get(mission, mission)

    features = {}

    # ============================================================
    # OBJETOS FUSIONADOS
    # ============================================================

    objects = fusion.get(
        "objects",
        pd.DataFrame(),
    )

    if not isinstance(objects, pd.DataFrame):
        objects = pd.DataFrame(objects)

    # ============================================================
    # FEATURES GERAIS DA FUSÃO
    # ============================================================

    features["vehicle_count"] = int(
        fusion.get(
            "object_count",
            len(objects),
        )
    )

    features["mean_confidence"] = float(
        fusion.get(
            "mean_confidence",
            0.0,
        )
    )

    features["inconsistencies"] = int(
        fusion.get(
            "inconsistencies",
            0,
        )
    )

    # ============================================================
    # VELOCIDADE
    # ============================================================

    if "speed_kmh" in objects.columns:

        speed = pd.to_numeric(
            objects["speed_kmh"],
            errors="coerce",
        ).dropna()

        if not speed.empty:

            features["mean_speed_kmh"] = float(
                speed.mean()
            )

            features["max_speed_kmh"] = float(
                speed.max()
            )

            features["min_speed_kmh"] = float(
                speed.min()
            )

            features["std_speed_kmh"] = float(
                speed.std()
            ) if len(speed) > 1 else 0.0

            features["stopped_count"] = int(
                (speed < 2.0).sum()
            )

            features["stopped_pct"] = float(
                (speed < 2.0).mean()
            )

        else:

            features["mean_speed_kmh"] = 0.0
            features["max_speed_kmh"] = 0.0
            features["min_speed_kmh"] = 0.0
            features["std_speed_kmh"] = 0.0
            features["stopped_count"] = 0
            features["stopped_pct"] = 0.0

    else:

        features["mean_speed_kmh"] = 0.0
        features["max_speed_kmh"] = 0.0
        features["min_speed_kmh"] = 0.0
        features["std_speed_kmh"] = 0.0
        features["stopped_count"] = 0
        features["stopped_pct"] = 0.0

    # ============================================================
    # ACELERAÇÃO
    # ============================================================

    if "acceleration_m_s2" in objects.columns:

        acceleration = pd.to_numeric(
            objects["acceleration_m_s2"],
            errors="coerce",
        ).dropna()

        if not acceleration.empty:

            features["mean_acceleration_m_s2"] = float(
                acceleration.mean()
            )

            features["max_acceleration_m_s2"] = float(
                acceleration.max()
            )

            features["min_acceleration_m_s2"] = float(
                acceleration.min()
            )

            features["hard_braking_count"] = int(
                (acceleration < -3.0).sum()
            )

        else:

            features["mean_acceleration_m_s2"] = 0.0
            features["max_acceleration_m_s2"] = 0.0
            features["min_acceleration_m_s2"] = 0.0
            features["hard_braking_count"] = 0

    else:

        features["mean_acceleration_m_s2"] = 0.0
        features["max_acceleration_m_s2"] = 0.0
        features["min_acceleration_m_s2"] = 0.0
        features["hard_braking_count"] = 0

    # ============================================================
    # QUALIDADE DA FUSÃO
    # ============================================================

    quality = fusion.get(
        "quality",
        {},
    )

    features["spatial_quality"] = float(
        quality.get(
            "spatial",
            0.0,
        )
    )

    features["temporal_quality"] = float(
        quality.get(
            "temporal",
            0.0,
        )
    )

    features["physical_quality"] = float(
        quality.get(
            "physical",
            0.0,
        )
    )

    features["fusion_quality"] = float(
        quality.get(
            "final",
            features["mean_confidence"],
        )
    )

    # ============================================================
    # MISSÃO 1 — AGRO
    # ============================================================

    if mission == "agro":

        source = sources.get("Sensores agrícolas")
        if source is None:
            source = sources.get("Agro")

        if source is not None and not source.empty:

            if "temperature_c" in source.columns:

                temperature = pd.to_numeric(
                    source["temperature_c"],
                    errors="coerce",
                ).dropna()

                if not temperature.empty:

                    features["mean_temperature_c"] = float(
                        temperature.mean()
                    )

                    features["max_temperature_c"] = float(
                        temperature.max()
                    )

            if "humidity_pct" in source.columns:

                humidity = pd.to_numeric(
                    source["humidity_pct"],
                    errors="coerce",
                ).dropna()

                if not humidity.empty:

                    features["mean_humidity_pct"] = float(
                        humidity.mean()
                    )

            if "fuel_l_h" in source.columns:
                fuel = pd.to_numeric(
                    source["fuel_l_h"],
                    errors="coerce",
                ).dropna()

                if not fuel.empty:
                    features["mean_fuel_l_h"] = float(fuel.mean())
                    features["max_fuel_l_h"] = float(fuel.max())

            elif "fuel_pct" in source.columns:
                fuel = pd.to_numeric(
                    source["fuel_pct"],
                    errors="coerce",
                ).dropna()

                if not fuel.empty:
                    features["mean_fuel_pct"] = float(fuel.mean())

    # ============================================================
    # MISSÃO 2 — ECONOMIA CIRCULAR
    # ============================================================

    elif mission == "circular":

        source = sources.get("BMS")

        if source is None:
            source = sources.get("Battery")

        if source is None:
            source = sources.get("Sistema BMS")

        if source is not None and not source.empty:

            if "temperature_c" in source.columns:

                temperature = pd.to_numeric(
                    source["temperature_c"],
                    errors="coerce",
                ).dropna()

                if not temperature.empty:

                    features["mean_battery_temperature_c"] = float(
                        temperature.mean()
                    )

                    features["max_battery_temperature_c"] = float(
                        temperature.max()
                    )

            if "internal_resistance_mohm" in source.columns:

                resistance = pd.to_numeric(
                    source["internal_resistance_mohm"],
                    errors="coerce",
                ).dropna()

                if not resistance.empty:

                    features["mean_internal_resistance_mohm"] = float(
                        resistance.mean()
                    )

                    features["max_internal_resistance_mohm"] = float(
                        resistance.max()
                    )

                    features["high_resistance_count"] = int(
                        (resistance > 24).sum()
                    )

            if "cycles" in source.columns:
                cycles = pd.to_numeric(
                    source["cycles"],
                    errors="coerce",
                ).dropna()

                if not cycles.empty:
                    features["mean_cycles"] = float(cycles.mean())
                    features["max_cycles"] = float(cycles.max())
                    features["high_cycle_count"] = int((cycles > 1200).sum())

            if "soc_pct" in source.columns:

                soc = pd.to_numeric(
                    source["soc_pct"],
                    errors="coerce",
                ).dropna()

                if not soc.empty:

                    features["mean_soc_pct"] = float(
                        soc.mean()
                    )

                    features["min_soc_pct"] = float(
                        soc.min()
                    )

    # ============================================================
    # MISSÃO 3 — LEAN FARM-TO-MARKET
    # ============================================================

    elif mission == "farm_to_market":

        source = sources.get("Logística")

        if source is None:
            source = sources.get("Logistica")

        if source is None:
            source = sources.get("Logistics")

        if source is not None and not source.empty:

            if "temperature_c" in source.columns:

                temperature = pd.to_numeric(
                    source["temperature_c"],
                    errors="coerce",
                ).dropna()

                if not temperature.empty:

                    features["mean_cargo_temperature_c"] = float(
                        temperature.mean()
                    )

                    features["max_cargo_temperature_c"] = float(
                        temperature.max()
                    )

            if "travel_time_min" in source.columns:

                travel_time = pd.to_numeric(
                    source["travel_time_min"],
                    errors="coerce",
                ).dropna()

                if not travel_time.empty:

                    features["mean_travel_time_min"] = float(
                        travel_time.mean()
                    )

                    features["max_travel_time_min"] = float(
                        travel_time.max()
                    )

            if "traffic_index" in source.columns:

                traffic = pd.to_numeric(
                    source["traffic_index"],
                    errors="coerce",
                ).dropna()

                if not traffic.empty:

                    features["mean_traffic_index"] = float(
                        traffic.mean()
                    )

                    features["max_traffic_index"] = float(
                        traffic.max()
                    )

            if "loading_delay_min" in source.columns:

                delay = pd.to_numeric(
                    source["loading_delay_min"],
                    errors="coerce",
                ).dropna()

                if not delay.empty:

                    features["mean_loading_delay_min"] = float(
                        delay.mean()
                    )

                    features["max_loading_delay_min"] = float(
                        delay.max()
                    )

    # ============================================================
    # MISSÃO 4 — MOBILIDADE
    # ============================================================

    elif mission == "mobility":

        source = sources.get("Visão urbana")

        if source is None:
            source = sources.get("Visao urbana")

        if source is None:
            source = sources.get("Urban Vision")

        if source is not None and not source.empty:

            if "distance_m" in source.columns:

                distance = pd.to_numeric(
                    source["distance_m"],
                    errors="coerce",
                ).dropna()

                if not distance.empty:

                    features["min_detected_distance_m"] = float(
                        distance.min()
                    )

                    features["mean_detected_distance_m"] = float(
                        distance.mean()
                    )

            if "crossing_active" in source.columns:

                crossing = source["crossing_active"]

                features["crossing_active_count"] = int(
                    crossing.astype(bool).sum()
                )

    # ============================================================
    # RETORNO
    # ============================================================

    return pd.DataFrame(
        [
            features
        ]
    )