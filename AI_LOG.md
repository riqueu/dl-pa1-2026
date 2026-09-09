# Registro de Uso de Ferramentas de IA (AI_LOG)

Documentação sintética do uso de ferramentas de IA no PA1, servindo como indicador geral dos momentos e frentes em que a assistência de IA foi integrada ao desenvolvimento.

---

## Episódio 1: Planejamento Geral do Projeto e Divisão de Tarefas
* **Contexto:** Estruturar a estratégia global do projeto (Partes 0, 1, 2 e ablações), alinhar os requisitos do edital (regras de autoria e proibições) e organizar o fluxo de trabalho colaborativo em dupla.
* **Uso da IA:** Auxiliou no desenho da arquitetura modular do repositório, na padronização prévia dos contratos de tensores (`image`, `mask_semantic`, `mask_instance`, `mask_three_class`) e na divisão das frentes de trabalho em branches de feature paralelas para evitar conflitos de merge.
* **Decisão Técnica:** Desenvolvimento modular e desacoplado entre os membros (Dados/Métricas vs. Modelagem/Pipelines).

---

## Episódio 2: Desenvolvimento dos Módulos Principais
* **Contexto:** Implementação dos datasets (sintético procedural e DSB2018 estratificado), U-Net com encoder pré-treinado, funções de perda customizadas e métricas autorais de matching 1-para-1.
* **Uso da IA:** Atuou como suporte de pair programming na escrita, vetorização e testes dos módulos em `src/` (`dataset.py`, `metrics.py`, `models.py`, `losses.py`, `postprocess.py`), assegurando o cumprimento estrito das restrições do edital.
* **Decisão Técnica:** Implementação 100% autoral das perdas, do decoder U-Net e dos algoritmos de matching (Hungarian e Greedy IoU), sem uso de bibliotecas prontas de detecção.

---

## Episódio 3: Execução, Avaliação e Validação Técnica do Pipeline
* **Contexto:** Treinamento das etapas iniciais (teste sintético e baseline semântico no DSB2018), validação de reprodutibilidade e quantificação empírica das limitações da abordagem ingênua por componentes conexos.
* **Uso da IA:** Apoio na estruturação dos pipelines de treino e avaliação em lote (`train.py`, `evaluate.py`), na formulação dos gráficos de diagnóstico (mAP vs. densidade de núcleos) e na montagem do caderno central de inferência (`inferencia.ipynb`) como vitrine técnica do projeto.
* **Decisão Técnica:** Fixação dos splits estratificados (`data/splits.json`) e escolha formal da Trilha A (Fronteiras e Watershed) para a resolução do colapso em aglomerados densos.
