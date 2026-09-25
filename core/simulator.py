import numpy as np
import pandas as pd

RNG = np.random.default_rng()


def _base(n, mode):
    if mode == "normal":
        speed = RNG.normal(42, 3.5, n).clip(8, 65)
        acc = RNG.normal(0, 0.25, n).clip(-1.5, 1.5)

    elif mode == "congestion":
        speed = RNG.normal(22, 5, n).clip(2, 45)
        acc = RNG.normal(-0.15, 0.65, n).clip(-3.5, 2)
        # Congestion: more stopped/very slow vehicles.
        stopped = max(4, n // 3)
        speed[:stopped] = RNG.uniform(2, 8, stopped)

    else:
        speed = RNG.normal(32, 6, n).clip(5, 60)
        acc = RNG.normal(-0.35, 0.9, n).clip(-5, 2)

        # Braking event distributed through several objects.
        count = max(5, n // 4)
        acc[:count] = RNG.uniform(-5, -3, count)
        speed[:count] = RNG.uniform(8, 25, count)

    return pd.DataFrame(
        {
            "object_id": [f"OBJ_{i:03d}" for i in range(n)],
            "timestamp": RNG.uniform(10, 20, n),
            "x": RNG.uniform(100, 900, n),
            "y": RNG.uniform(80, 520, n),
            "speed_kmh": speed,
            "acceleration_m_s2": acc,
            "heading_deg": RNG.uniform(0, 360, n),
        }
    )


def generate_scenario(mission, scenario):
    """
    Gera cenários sintéticos coerentes para o protótipo.

    A nomenclatura dos cenários é mantida compatível com o restante
    do protótipo: cenário normal = Fluxo normal; cenários críticos
    possuem alterações observáveis nas fontes.
    """

    scenario_mode = {
        "Fluxo normal": "normal",
        "Condição operacional anormal": "braking",
        "Uso severo": "braking",
        "Degradação de componente": "braking",
        "Atraso logístico": "congestion",
        "Congestionamento": "congestion",
        "Frenagem brusca": "braking",
        "Conflito veículo × pedestre": "braking",
        "Falha de sensor": "normal",
    }

    mode = scenario_mode.get(scenario, "normal")
    n = 24 if mode == "normal" else 32
    base = _base(n, mode)

    def src(name, dx, dy, ds, conf):
        d = base.copy()
        d.x += RNG.normal(0, dx, n)
        d.y += RNG.normal(0, dy, n)
        d.speed_kmh += RNG.normal(0, ds, n)
        d.timestamp += RNG.normal(0, 0.06, n)
        d["source"] = name
        d["confidence"] = RNG.uniform(*conf, n)
        return d

    v2x = base.copy()
    v2x["source"] = "V2X"
    v2x["confidence"] = RNG.uniform(0.95, 0.99, n)

    rsu = src("RSU", 2.5, 2.5, 1.0, (0.90, 0.97))
    vision = src("Visão", 4, 4, 1.8, (0.82, 0.94))

    # Falha de sensor: divergência real em uma parcela das observações.
    if scenario == "Falha de sensor":
        count = max(4, n // 4)
        vision.loc[: count - 1, "speed_kmh"] += RNG.uniform(28, 40, count)
        vision.loc[: count - 1, "confidence"] = RNG.uniform(0.35, 0.55, count)

    sources = {
        "V2X": v2x,
        "RSU": rsu,
        "Visão": vision,
    }

    # ============================================================
    # AGRO
    # ============================================================
    if mission == "Agro Inteligente":
        d = pd.DataFrame(
            {
                "object_id": [f"MACH_{i:02d}" for i in range(8)],
                "timestamp": RNG.uniform(10, 20, 8),
                "speed_kmh": RNG.normal(8, 1.2, 8).clip(2, 14),
                "temperature_c": RNG.normal(72, 4, 8),
                "humidity_pct": RNG.normal(68, 6, 8),
                "fuel_l_h": RNG.normal(14, 1.2, 8),
                "confidence": RNG.uniform(0.88, 0.97, 8),
                "source": "Sensores agrícolas",
            }
        )

        if scenario == "Condição operacional anormal":
            count = 3
            d.loc[: count - 1, "temperature_c"] = RNG.uniform(90, 98, count)
            d.loc[: count - 1, "fuel_l_h"] = RNG.uniform(18, 21, count)

        sources["Sensores agrícolas"] = d

    # ============================================================
    # ECONOMIA CIRCULAR
    # ============================================================
    elif mission == "Economia Circular":
        d = pd.DataFrame(
            {
                "object_id": [f"BAT_{i:02d}" for i in range(10)],
                "timestamp": RNG.uniform(10, 20, 10),
                "soc_pct": RNG.normal(66, 9, 10).clip(15, 95),
                "temperature_c": RNG.normal(34, 3.5, 10),
                "cycles": RNG.integers(300, 1100, 10),
                "internal_resistance_mohm": RNG.normal(18, 2.5, 10).clip(8, 30),
                "confidence": RNG.uniform(0.91, 0.98, 10),
                "source": "BMS",
            }
        )

        if scenario in ("Degradação de componente", "Uso severo"):
            count = 4
            d.loc[: count - 1, "temperature_c"] = RNG.uniform(46, 53, count)
            d.loc[: count - 1, "internal_resistance_mohm"] = RNG.uniform(26, 31, count)
            d.loc[: count - 1, "cycles"] = RNG.integers(1200, 1600, count)

        sources["BMS"] = d

    # ============================================================
    # FARM-TO-MARKET
    # ============================================================
    elif mission == "Lean Farm-to-Market":
        d = pd.DataFrame(
            {
                "shipment_id": [f"LOTE_{i:03d}" for i in range(10)],
                "timestamp": RNG.uniform(10, 20, 10),
                "temperature_c": RNG.normal(6, 0.8, 10),
                "travel_time_min": RNG.normal(180, 18, 10),
                "traffic_index": RNG.uniform(0.2, 0.65, 10),
                "loading_delay_min": RNG.normal(12, 4, 10).clip(0, 30),
                "confidence": RNG.uniform(0.88, 0.98, 10),
                "source": "Logística",
            }
        )

        if scenario == "Atraso logístico":
            count = 4
            d.loc[: count - 1, "traffic_index"] = RNG.uniform(0.85, 0.99, count)
            d.loc[: count - 1, "travel_time_min"] = RNG.uniform(235, 285, count)
            d.loc[: count - 1, "loading_delay_min"] = RNG.uniform(25, 40, count)

        sources["Logística"] = d

    # ============================================================
    # MOBILIDADE
    # ============================================================
    elif mission == "Mobilidade para Pessoas":
        d = pd.DataFrame(
            {
                "object_id": [f"PERSON_{i:02d}" for i in range(8)],
                "timestamp": RNG.uniform(10, 20, 8),
                "distance_m": RNG.uniform(20, 100, 8),
                "crossing_active": RNG.choice([0, 1], 8, p=[0.7, 0.3]),
                "confidence": RNG.uniform(0.86, 0.98, 8),
                "source": "Visão urbana",
            }
        )

        if scenario == "Conflito veículo × pedestre":
            count = 4
            d.loc[: count - 1, "distance_m"] = RNG.uniform(8, 18, count)
            d.loc[: count - 1, "crossing_active"] = 1

        sources["Visão urbana"] = d

    return {"sources": sources}
