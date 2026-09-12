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
* **Uso da IA:** Atuou como suporte de pair programming na escrita, vetorização e testes dos módulos em `src/` (`dataset.py`, `metrics.py`, `models.py`, `losses.py`, `postprocess.py`), incluindo a formulação da Trilha A (segmentação 3 classes e decodificação por watershed), assegurando o cumprimento estrito das restrições do edital.
* **Decisão Técnica:** Implementação 100% autoral das perdas (ponderadas e focais), do decoder U-Net, da decodificação watershed e dos algoritmos de matching (Hungarian e Greedy IoU), sem uso de bibliotecas prontas de detecção.

---

## Episódio 3: Execução, Avaliação e Validação Técnica do Pipeline
* **Contexto:** Treinamento das etapas iniciais (teste sintético e baseline semântico no DSB2018), validação de reprodutibilidade e quantificação empírica das limitações da abordagem ingênua por componentes conexos.
* **Uso da IA:** Apoio na estruturação dos pipelines de treino e avaliação em lote (`train.py`, `evaluate.py`), na formulação dos gráficos de diagnóstico e na montagem do caderno central de inferência (`inferencia.ipynb`) como vitrine técnica do projeto.
* **Decisão Técnica:** Fixação dos splits estratificados (`data/splits.json`), escolha da Trilha A e otimização vetorizada do cálculo de IoU pareado por histograma.

---

## Episódio 4: Ablação de Funções de Perda
* **Contexto:** Comparar CE ponderada e focal multiclasse com $\gamma \in \{0,1,2,5\}$ usando duas seeds por configuração.
* **Uso da IA:** Apoio na automação da grade experimental, consolidação das métricas e geração do gráfico da ablação.
* **Decisão Técnica:** Manter arquitetura, split e pós-processamento fixos e identificar $\gamma=0$ como o melhor resultado observado para a decodificação por watershed.

---

## Episódio 5: Ablação Arquitetural com DeepLab e ASPP
* **Contexto:** Implementar o Eixo 1 da Parte 3 sem confundir um ASPP isolado no gargalo com o mecanismo completo de preservação de resolução descrito no enunciado.
* **Uso da IA:** Apoio na revisão do plano, na implementação pareada do encoder ResNet34 com *output stride* 16, do ASPP autoral e dos contratos de checkpoint, além da criação de testes de regressão e automação das duas seeds.
* **Decisão Técnica:** A variante DeepLab remove o stride do `layer4`, aplica dilatação 2 e usa ASPP com taxas 6, 12 e 18. A U-Net permanece inalterada por padrão, garantindo a recarga estrita dos checkpoints anteriores.

---

## Episódio 6: Inferência em Mosaico e Diagnóstico de Falha em Bordas
* **Contexto:** Montagem de imagens grandes ($512 \times 512$), inferência em janelas deslizantes sobrepostas ($256 \times 256$, stride 128) e diagnóstico da quebra de instâncias na fronteira sem fusão.
* **Uso da IA:** Auxílio na implementação do módulo `src/mosaic.py`, automação do pipeline de teste em grade e lâminas contínuas (`scripts/run_mosaic_demo.py`), diagnóstico geométrico de fatiamento de núcleos e geração dos plots com zoom.
* **Decisão Técnica:** Desacoplamento estrito entre geração de tiles/predições e o algoritmo de costura do Membro B (`src/stitching.py`), com quantificação do colapso de mAP (queda de $\sim 0.5157$ para $0.3898$ em grade e $0.2079$ em lâmina contínua) e inflação de contagem por fragmentação.

---

## Episódio 7: Fusão de Instâncias em Inferência por Tiles
* **Contexto:** Corrigir a duplicação e fragmentação de núcleos quando cada tile sobreposto é decodificado independentemente por watershed.
* **Uso da IA:** Apoio no desenho e teste de uma costura baseada em matching local por IoU, Union-Find e arbitragem espacial dos pixels conflitantes.
* **Decisão Técnica:** Cada par `(tile, id_local)` recebe uma identidade provisória; correspondências um-para-um na faixa de sobreposição são unidas transitivamente e os IDs globais são renumerados. A versão sem fusão usa a mesma regra de composição para permitir uma comparação controlada antes/depois.
