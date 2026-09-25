# V2X-AI — Interactive Prototype

Protótipo interativo para demonstrar uma arquitetura de fusão multimodal e inteligência aplicada a quatro missões:

- Agro Inteligente
- Economia Circular
- Lean Farm-to-Market
- Mobilidade para Pessoas

## Arquitetura

Coleta → Fusão → Feature Engineering → Anomaly Detection → Machine Learning → Predição → Decision Layer

O protótipo utiliza dados sintéticos coerentes para demonstração pública. O SUMO é usado no desenvolvimento/validação e não é necessário para executar o aplicativo publicado.

## Execução local

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Publicação

O projeto foi estruturado para Streamlit Community Cloud. No GitHub, publique o conteúdo deste diretório e selecione `app.py` como arquivo principal.

## Observação técnica

A detecção de anomalias e a predição de risco são componentes heurísticos/prototípicos. Não representam um modelo validado com dados reais de produção.
