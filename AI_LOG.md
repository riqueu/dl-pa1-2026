# Registro de Uso de Ferramentas de IA (AI_LOG)

Documentação do uso de ferramentas de IA no PA1, detalhando decisões técnicas e tarefas desenvolvidas com assistência de IA.

---

## Episódio 1: Padronização de Formatos e Planejamento em Dupla (07/09/2026)
* **Contexto:** Alinhar os requisitos do `PA1.pdf` (proibição de detectores prontos, matching e decoder autorais) e estruturar o trabalho em dupla sem conflitos de merge.
* **Uso da IA:** Auxiliou na formulação da divisão de tarefas (Dados/Métricas vs. Modelagem/Treino) e na padronização prévia dos contratos de tensores (`image`, `mask_semantic`, `mask_instance` e assinaturas das métricas).
* **Decisão:** Desenvolvimento modular isolado por arquivos e branches de feature, garantindo compatibilidade entre os membros da dupla.

---

## Episódio 2: Implementação de Datasets e Métricas (08/09/2026)
* **Branch:** `feature/dataset-and-metrics` (`src/dataset.py`, `src/metrics.py`).
* **Uso da IA:**
  - **Sintético (Parte 0):** Implementação procedural com NumPy para desenhar 5 a 20 elipses rotacionadas sobrepostas com ruído e contraste em $128 \times 128$.
  - **Métricas (Partes 0 e 1):** Implementação do matching 1-para-1 (Hungarian e Greedy IoU) para mAP@[.50:.95] e erro de contagem, além de IoU e Dice semânticos.
  - **DSB2018 (Parte 1):** Loader da base real com fusão de máscaras via interpolação Nearest Neighbor e função de estratificação treino/val/teste por modalidade visual de microscopia.
* **Decisão:** Cumprimento estrito das regras de engenharia (sem bibliotecas de detecção externas) e validação dos módulos via testes unitários automatizados.