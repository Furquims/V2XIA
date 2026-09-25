
import math

def _mission_id(mission):
    return {
        "Agro Inteligente": "agro",
        "Economia Circular": "circular",
        "Lean Farm-to-Market": "farm_to_market",
        "Mobilidade para Pessoas": "mobility",
        "agro": "agro",
        "circular": "circular",
        "farm_to_market": "farm_to_market",
        "mobility": "mobility",
    }.get(mission, mission)

def _row(features):
    if hasattr(features, "iloc"):
        return features.iloc[0].to_dict() if not features.empty else {}
    return features if isinstance(features, dict) else {}

def _num(row, key, default=0.0):
    try:
        v = row.get(key, default)
        return float(default if v is None else v)
    except (TypeError, ValueError):
        return float(default)

def _anomaly_score(anomaly):
    try:
        return float(anomaly.get("score", 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0

def _active_anomaly(anomaly):
    if not isinstance(anomaly, dict):
        return []
    signals = anomaly.get("signals", {}) or {}
    active = []
    if signals.get("inconsistency_signal", 0) > 0.05:
        active.append("inconsistência entre fontes")
    if signals.get("confidence_signal", 0) > 0.05:
        active.append("baixa confiança da fusão")
    if signals.get("braking_signal", 0) > 0.05:
        active.append("frenagem brusca")
    if signals.get("acceleration_signal", 0) > 0.05:
        active.append("aceleração extrema")
    return active

def build_decision(mission, scenario, features, prediction, anomaly):
    row = _row(features)
    mid = _mission_id(mission)

    risk = prediction.get("label", "Baixo") if isinstance(prediction, dict) else "Baixo"
    score = float(prediction.get("score", 0.0)) if isinstance(prediction, dict) else 0.0
    anomaly_score = _anomaly_score(anomaly)
    signals = (anomaly or {}).get("signals", {}) if isinstance(anomaly, dict) else {}

    confidence = _num(row, "mean_confidence", 1.0)
    inconsistencies = int(round(_num(row, "inconsistencies")))
    hard_braking = int(round(_num(row, "hard_braking_count")))
    stopped = int(round(_num(row, "stopped_count")))

    # Contexto específico da missão.
    if mid == "agro":
        temp = _num(row, "max_temperature_c")
        fuel = _num(row, "mean_fuel_l_h")
        if temp >= 88 or fuel >= 18:
            event = "Desvio operacional no equipamento agrícola"
            action = "Reduzir a carga operacional do equipamento, inspecionar o sistema térmico e verificar consumo de combustível antes de retornar à operação normal."
            reason = f"Temperatura máxima de {temp:.1f} °C e consumo médio de {fuel:.1f} L/h indicam condição operacional fora da faixa de referência."
            impact = "Evitar degradação do equipamento, parada não planejada e consumo excessivo durante a operação."
        elif risk == "Moderado":
            event = "Atenção operacional no equipamento agrícola"
            action = "Acompanhar temperatura e consumo em tempo real e programar inspeção caso a tendência de elevação continue."
            reason = f"Os indicadores apresentam desvio moderado: {temp:.1f} °C de temperatura máxima e {fuel:.1f} L/h de consumo."
            impact = "Reduzir a probabilidade de falha e manter produtividade com intervenção antecipada."
        else:
            event = "Operação agrícola dentro do padrão"
            action = "Manter operação e continuar o monitoramento de temperatura, consumo e comportamento do equipamento."
            reason = "Os indicadores operacionais permanecem próximos da condição esperada."
            impact = "Continuidade operacional e detecção antecipada de desvios."

    elif mid == "circular":
        temp = _num(row, "max_battery_temperature_c")
        resistance = _num(row, "max_internal_resistance_mohm")
        cycles = _num(row, "max_cycles")
        if temp >= 45 or resistance >= 26 or cycles >= 1400:
            event = "Indício de degradação do componente"
            action = "Retirar o componente da operação severa, executar diagnóstico de saúde e registrar o estado no histórico para definir manutenção, recondicionamento ou substituição."
            reason = f"Temperatura máxima de {temp:.1f} °C, resistência interna máxima de {resistance:.1f} mΩ e {cycles:.0f} ciclos indicam sinais de envelhecimento."
            impact = "Prolongar a vida útil do componente, reduzir descarte prematuro e evitar falha durante a operação."
        elif risk == "Moderado":
            event = "Acompanhamento de degradação"
            action = "Aumentar a frequência de monitoramento do componente e comparar os indicadores com seu histórico de utilização."
            reason = f"Os indicadores mostram tendência de degradação sem evidência suficiente para retirada imediata."
            impact = "Antecipação da manutenção e melhor aproveitamento da vida útil."
        else:
            event = "Estado do componente dentro do padrão"
            action = "Manter o componente em operação e continuar registrando temperatura, resistência e histórico de ciclos."
            reason = "Não foram observados indicadores relevantes de degradação no cenário atual."
            impact = "Maior rastreabilidade e uso orientado à vida útil."

    elif mid == "farm_to_market":
        traffic = _num(row, "max_traffic_index")
        travel = _num(row, "max_travel_time_min")
        delay = _num(row, "max_loading_delay_min")
        if traffic >= .85 or travel >= 230 or delay >= 25:
            event = "Gargalo logístico identificado"
            action = "Replanejar a rota ou janela de entrega, priorizar a carga afetada e verificar a origem do atraso no trecho ou no processo de carregamento."
            reason = f"Índice de tráfego {traffic:.2f}, tempo de viagem de {travel:.0f} min e atraso de carregamento de {delay:.0f} min indicam perda de eficiência logística."
            impact = "Reduzir tempo de transporte, espera e risco de perda ou atraso da carga."
        elif risk == "Moderado":
            event = "Atenção à eficiência logística"
            action = "Acompanhar a rota e o tempo estimado de chegada e preparar uma alternativa caso o atraso continue aumentando."
            reason = "Há sinais de deterioração do fluxo logístico, mas ainda sem necessidade de intervenção imediata."
            impact = "Redução de atrasos e melhor previsibilidade da entrega."
        else:
            event = "Fluxo logístico dentro do padrão"
            action = "Manter rota e janela de entrega atuais, continuando o monitoramento de tráfego e tempo de viagem."
            reason = "Os indicadores logísticos permanecem dentro da condição esperada."
            impact = "Previsibilidade e continuidade do fluxo farm-to-market."

    elif mid == "mobility":
        distance = _num(row, "min_detected_distance_m", 999)
        crossing = _num(row, "crossing_active_count")
        if distance <= 18 and crossing > 0:
            event = "Conflito potencial entre veículo e pedestre"
            action = "Reduzir a velocidade do veículo na zona de conflito e acionar alerta ao sistema de mobilidade para priorizar a travessia e acompanhar a aproximação."
            reason = f"Foi detectada travessia ativa com distância mínima de {distance:.1f} m entre o veículo e o pedestre."
            impact = "Reduzir a exposição ao risco e aumentar a segurança de pedestres e ocupantes."
        elif hard_braking >= 5:
            event = "Evento de frenagem brusca"
            action = "Alertar o entorno sobre a desaceleração e analisar o trecho para verificar recorrência de eventos de frenagem."
            reason = f"Foram identificadas {hard_braking} frenagens bruscas no conjunto observado."
            impact = "Reduzir risco de colisões secundárias e identificar pontos de risco recorrente."
        elif risk == "Moderado":
            event = "Condição de mobilidade requer atenção"
            action = "Manter monitoramento reforçado de veículos, obstáculos e usuários vulneráveis no entorno."
            reason = "O conjunto de sinais apresenta risco moderado e requer acompanhamento."
            impact = "Antecipação de situações de risco sem intervenção desnecessária."
        else:
            event = "Mobilidade dentro do padrão"
            action = "Manter monitoramento do ambiente e compartilhamento das informações relevantes entre veículos e infraestrutura."
            reason = "Não foram identificados sinais relevantes de conflito ou degradação operacional."
            impact = "Maior previsibilidade e suporte à mobilidade segura."

    else:
        event = "Condição operacional monitorada"
        action = "Acompanhar os indicadores e atuar sobre os sinais que apresentarem tendência de deterioração."
        reason = "A camada de inteligência identificou o estado atual a partir dos dados fundidos."
        impact = "Redução de riscos e melhoria da eficiência operacional."

    # Prioridade final: risco e anomalia elevam a urgência, mas a ação permanece contextual.
    if risk == "Alto" or anomaly_score >= 0.45:
        priority = "Alta"
    elif risk == "Moderado" or anomaly_score >= 0.20:
        priority = "Média"
    else:
        priority = "Baixa"

    anomaly_text = ""
    if anomaly_score >= 0.20:
        active = _active_anomaly(anomaly)
        if active:
            anomaly_text = " Sinais adicionais: " + ", ".join(active) + "."
        else:
            anomaly_text = " O detector de anomalias também identificou desvio no comportamento observado."

    justification = (
        f"{reason} Risco {risk}, score {score:.2f}, confiança média de "
        f"{confidence:.1%}, {inconsistencies} inconsistência(s) e "
        f"{hard_braking} frenagem(ns) brusca(s). "
        f"Score de anomalia: {anomaly_score:.2f}.{anomaly_text}"
    )

    return {
        "event": event,
        "priority": priority,
        "action": action,
        "justification": justification,
        "impact": impact,
        "mission": mission,
        "scenario": scenario,
        "evidence": {
            "risk_score": round(score, 3),
            "anomaly_score": round(anomaly_score, 3),
            "mean_confidence": round(confidence, 3),
            "inconsistencies": inconsistencies,
            "hard_braking_count": hard_braking,
            "stopped_count": stopped,
        },
    }
