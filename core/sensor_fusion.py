import numpy as np
import pandas as pd


def _quality(a, b, c):
    sd = lambda x, y: np.sqrt((x["x"] - y["x"]) ** 2 + (x["y"] - y["y"]) ** 2)

    spatial = float(
        np.clip(
            1 - np.mean([sd(a, b), sd(a, c)]) / 25,
            0,
            1,
        )
    )

    temporal = float(
        np.clip(
            1 - np.mean(
                [
                    abs(a["timestamp"] - b["timestamp"]),
                    abs(a["timestamp"] - c["timestamp"]),
                ]
            )
            / 0.5,
            0,
            1,
        )
    )

    physical = float(
        np.clip(
            1
            - np.mean(
                [
                    abs(a["speed_kmh"] - b["speed_kmh"]),
                    abs(a["speed_kmh"] - c["speed_kmh"]),
                ]
            )
            / 25,
            0,
            1,
        )
    )

    return spatial, temporal, physical


def _weighted(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return float(np.sum(values * weights) / total) if total > 0 else float(np.mean(values))


def fuse_observations(sources):
    base, rsu, vision = [
        sources[k].reset_index(drop=True)
        for k in ("V2X", "RSU", "Visão")
    ]

    rows = []
    assoc = []
    bad = 0

    for i in range(len(base)):
        cand = [
            ("V2X", base.iloc[i]),
            ("RSU", rsu.iloc[i]),
            ("Visão", vision.iloc[i]),
        ]

        speeds = np.array([x[1]["speed_kmh"] for x in cand], dtype=float)
        conf = np.array([x[1]["confidence"] for x in cand], dtype=float)

        median_speed = np.median(speeds)
        adj = conf.copy()

        for j, value in enumerate(speeds):
            deviation = abs(value - median_speed)
            if deviation > 10:
                adj[j] *= 0.25
            elif deviation > 5:
                adj[j] *= 0.60

        inconsistent = bool(speeds.max() - speeds.min() > 15)
        bad += int(inconsistent)

        weights = adj / adj.sum()

        x = np.array([z[1]["x"] for z in cand], dtype=float)
        y = np.array([z[1]["y"] for z in cand], dtype=float)
        t = np.array([z[1]["timestamp"] for z in cand], dtype=float)

        spatial, temporal, physical = _quality(
            base.iloc[i],
            rsu.iloc[i],
            vision.iloc[i],
        )

        final = float(
            np.clip(
                0.35 * np.sum(adj * weights)
                + 0.25 * spatial
                + 0.20 * temporal
                + 0.20 * physical,
                0.05,
                0.99,
            )
        )

        row = {
            "fused_object_id": f"FUSED_{base.iloc[i].object_id}",
            "x": round(np.sum(x * weights), 2),
            "y": round(np.sum(y * weights), 2),
            "timestamp": round(np.sum(t * weights), 3),
            "speed_kmh": round(np.sum(speeds * weights), 2),
            "confidence": round(final, 3),
            "sources": "V2X + RSU + Visão",
            "inconsistent": "Sim" if inconsistent else "Não",
        }

        # Preserva a variável de aceleração para a camada de inteligência.
        # O sensor fusion anterior descartava esta informação, fazendo
        # hard_braking_count e min_acceleration_m_s2 ficarem sempre zerados.
        if all("acceleration_m_s2" in z[1] for z in cand):
            accelerations = np.array(
                [z[1]["acceleration_m_s2"] for z in cand],
                dtype=float,
            )
            row["acceleration_m_s2"] = round(
                _weighted(accelerations, weights),
                3,
            )

        rows.append(row)

        assoc += [
            {
                "Fused object": f"FUSED_{base.iloc[i].object_id}",
                "Source": name,
                "Source object": str(record.object_id),
                "Speed (km/h)": round(float(record.speed_kmh), 2),
                "Initial confidence": round(float(record.confidence), 3),
                "Adjusted weight": round(float(weights[j]), 3),
            }
            for j, (name, record) in enumerate(cand)
        ]

    objects = pd.DataFrame(rows)
    association = pd.DataFrame(assoc)

    spatial, temporal, physical = _quality(
        base.iloc[0],
        rsu.iloc[0],
        vision.iloc[0],
    )

    return {
        "objects": objects,
        "observation_count": sum(len(d) for d in sources.values()),
        "object_count": len(objects),
        "mean_confidence": float(objects.confidence.mean()),
        "inconsistencies": bad,
        "quality": {
            "spatial": spatial,
            "temporal": temporal,
            "physical": physical,
            "final": float(objects.confidence.mean()),
        },
        "association_sample": association.head(15),
    }
