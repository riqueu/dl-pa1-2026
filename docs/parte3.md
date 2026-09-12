# Plano de Implementação da Parte 3 — Ablações (Eixos 1 e 2)

Este documento estabelece o plano técnico, os contratos de código e a divisão de tarefas em paralelo para a execução da **Parte 3 — Ablações** do PA1 (Deep Learning — FGV EMAp).

Conforme estipulado no edital:
> *"Rodem ablações em dois dos eixos abaixo, cada configuração com 2 seeds, reportando média ± desvio: (...)"*

A dupla selecionou formalmente:
1. **Eixo 1 — Como Recuperar Resolução:** Comparação entre *Skip Connections* (U-Net), *Atrous Convolution + ASPP* (DeepLabv3) e baseline sem skips (*No-Skips*).
2. **Eixo 2 — A Função de Perda:** Avaliação do desbalanceamento severo da classe fronteira ($2.8\%$) variando $\text{CE Ponderada}$ vs. $\text{Focal Loss}$ com $\gamma \in \{0, 1, 2, 5\}$.

---

## 1. Estrutura das Ablações

### 1.1. Eixo 1: Mecanismos de Recuperação de Resolução
* **Objetivo:** Investigar o impacto de diferentes mecanismos arquiteturais para restaurar os detalhes finos e contornos de células no mesmo encoder (`resnet34`), mantendo a decodificação da Parte 2 (Watershed 3-classes).
* **Configurações a comparar (todas com 15 épocas):**
  1. `unet_skips`: U-Net padrão com concatenação de skip connections (resolução preservada estágio a estágio).
  2. `unet_noskips`: U-Net sem skip connections (`--no_skips`), onde o decoder opera puramente sobre o gargalo.
  3. `deeplab_aspp`: ResNet34 com *output stride* 16, `layer4` dilatado, ASPP com ramo $1\times1$, taxas $r \in \{6, 12, 18\}$ e pooling global, sem depender de skips rasos.
* **Seeds por configuração:** `seed=42` e `seed=123`.
* **Total de execuções:** $3 \text{ configurações} \times 2 \text{ seeds} = 6 \text{ treinos}$.
* **Sinergia:** O módulo ASPP e o cálculo do seu campo receptivo adiantam diretamente as respostas obrigatórias da **Parte 5** (Galeria de Falhas) e da **Parte 6** (Invariância a Escala).

---

### 1.2. Eixo 2: A Função de Perda e Foco em Classes Minoritárias
* **Objetivo:** A classe fronteira ocupa apenas $\approx 2.8\%$ dos pixels no DSB2018. Analisar o comportamento da rede ao transitar de entropia cruzada sem penalização até focal loss com foco agressivo em pixels difíceis.
* **Configurações a comparar (todas na U-Net ResNet34 com Watershed):**
  1. `ce_weighted`: Cross-Entropy multiclasse ponderada por pesos inversos suavizados (`[1.0, 2.9, 5.7]`).
  2. `focal_gamma0`: Focal Loss com $\gamma = 0$ (matematicamente equivalente à Cross-Entropy padrão).
  3. `focal_gamma1`: Focal Loss com $\gamma = 1$.
  4. `focal_gamma2`: Focal Loss com $\gamma = 2$ (foco moderado em pixels de difícil separação).
  5. `focal_gamma5`: Focal Loss com $\gamma = 5$ (foco extremo em pixels de fronteira/ambíguos).
* **Seeds por configuração:** `seed=42` e `seed=123`.
* **Total de execuções:** $5 \text{ configurações} \times 2 \text{ seeds} = 10 \text{ treinos}$.

---

## 2. Divisão de Tarefas da Dupla (Paralelismo Real)

A divisão mantém isolamento absoluto de arquivos para trabalho simultâneo em branches separadas:

```
                               ┌────────────────────────────────────────┐
                               │           main (Parte 2 Concluída)     │
                               └───────────────────┬────────────────────┘
                                                   │
                      ┌────────────────────────────┴────────────────────────────┐
                      ▼                                                         ▼
        feature/part3-eixo1-resolution                             feature/part3-eixo2-losses
        (Membro A - Isaias)                                        (Membro B - Henrique)
        • src/models.py (ASPP / DeepLab)                           • src/losses.py & train.py
        • Argumento --decoder [unet, aspp]                         • Script de automação da grade de perdas
        • Rodar 6 treinos (U-Net, No-Skips, ASPP)                  • Rodar 10 treinos (CE, Focal γ=0, 1, 2, 5)
        • Tabular mAP ± desvio e campo receptivo                   • Tabular mAP ± desvio e curvas de treino
                      │                                                         │
                      └────────────────────────────┬────────────────────────────┘
                                                   ▼
                                       Integração & Consolidação
                                       • docs/resultados_parte3.md
                                       • notebooks/inferencia.ipynb
```

### 2.1. Membro A (Isaias) — Eixo 1 (Recuperação de Resolução & ASPP)
* **Branch:** `feature/part3-eixo1-resolution`
* **Arquivos:** `src/models.py`, `train.py`
* **Passo a Passo:**
  1. **Implementar o bloco ASPP em `src/models.py`:**
     - Ramos paralelos: Conv $1\times 1$, Convs $3\times 3$ com dilatações $(6, 12, 18)$, e AdaptiveAvgPool $1\times 1$.
     - Projeção de fusão $1\times 1$, BatchNorm, Dropout e ReLU.
  2. **Integrar opção de Decoder em `build_model` / `train.py`:**
     - Adicionar argumento `--decoder_type` com opções `unet` (padrão) e `aspp`.
     - Quando `aspp`, acoplar o ASPP na saída do `layer4` do encoder e decodificar para a resolução de entrada.
  3. **Execução das 6 corridas (2 seeds cada):**
     * Config 1: U-Net com Skips (`seed=42`, `seed=123`)
     * Config 2: U-Net sem Skips (`--no_skips`, `seed=42`, `seed=123`)
     * Config 3: DeepLab ASPP (`--decoder_type aspp`, `seed=42`, `seed=123`)
  4. **Avaliação:** Rodar `evaluate.py` para cada checkpoint e salvar métricas em `outputs/part3_eixo1/`.
  5. **Cálculo Teórico:** Calcular o campo receptivo teórico com e sem atrous convolution (preparando a Parte 5).

---

### 2.2. Membro B (Henrique) — Eixo 2 (Funções de Perda & Fator $\gamma$)
* **Branch:** `feature/part3-eixo2-losses`
* **Arquivos:** `scripts/run_eixo2.sh` (ou Python runner), `outputs/part3_eixo2/`
* **Passo a Passo:**
  1. **Automatizar as 10 corridas da grade de perdas:**
     * `CE Ponderada`: `--loss weighted_ce_3c --class_weights auto` (seeds 42, 123)
     * `Focal γ=0`: `--loss multiclass_focal --gamma 0.0 --class_weights auto` (seeds 42, 123)
     * `Focal γ=1`: `--loss multiclass_focal --gamma 1.0 --class_weights auto` (seeds 42, 123)
     * `Focal γ=2`: `--loss multiclass_focal --gamma 2.0 --class_weights auto` (seeds 42, 123)
     * `Focal γ=5`: `--loss multiclass_focal --gamma 5.0 --class_weights auto` (seeds 42, 123)
  2. **Avaliação Padronizada:** Executar `evaluate.py` para os 10 checkpoints no split de validação (`val`) com Hungarian matching.
  3. **Tabulação:** Extrair as métricas de `metrics_summary.json` de cada execução e calcular:
     $$\text{mAP}_{\text{médio}} \pm \sigma, \quad \text{ErroContagem}_{\text{médio}} \pm \sigma$$
  4. **Análise Gráfica:** Gerar curva de $\text{mAP} \times \gamma$ e evolução do erro de contagem para a apresentação.

---

## 3. Matriz de Resultados Esperada (Template de Entrega)

### Tabela 1: Eixo 1 — Recuperação de Resolução
| Arquitetura | Mecanismo | Parâmetros | mAP@[.50:.95] (Seed 42) | mAP@[.50:.95] (Seed 123) | Média $\pm$ Desvio | Erro Médio Contagem |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| U-Net Padrão | Skip Connections | 25.834.003 | — | — | **0.XXX $\pm$ 0.0YY** | — |
| U-Net Sem Skips | Gargalo Direto | 25.041.427 | — | — | **0.XXX $\pm$ 0.0YY** | — |
| DeepLab/ASPP | Atrous + ASPP ($r=1,6,12,18$) | 25.711.555 | — | — | **0.XXX $\pm$ 0.0YY** | — |

### Tabela 2: Eixo 2 — Função de Perda
| Perda | Parâmetro $\gamma$ | Pesos de Classe | mAP (Seed 42) | mAP (Seed 123) | Média $\pm$ Desvio | Erro Médio Contagem |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CE Ponderada | — | Auto (`[1.0, 2.9, 5.7]`) | 0.5157 | — | **0.XXX $\pm$ 0.0YY** | — |
| Multiclass Focal | $\gamma = 0$ (CE Pura) | Auto | — | — | **0.XXX $\pm$ 0.0YY** | — |
| Multiclass Focal | $\gamma = 1$ | Auto | — | — | **0.XXX $\pm$ 0.0YY** | — |
| Multiclass Focal | $\gamma = 2$ | Auto | — | — | **0.XXX $\pm$ 0.0YY** | — |
| Multiclass Focal | $\gamma = 5$ | Auto | — | — | **0.XXX $\pm$ 0.0YY** | — |

---

## 4. Próximos Passos de Execução

1. Criar as branches `feature/part3-eixo1-resolution` e `feature/part3-eixo2-losses`.
2. O Membro A implementa o módulo ASPP em `src/models.py`.
3. O Membro B dispara os scripts de treinamento da grade de $\gamma$.
4. Avaliação e merge na `main` com consolidação no `notebooks/inferencia.ipynb`.

### Estado da implementação do Eixo 1

Implementado na branch `feature/part3-eixo1-resolution`:

- `ResNetEncoder` com *output stride* 16 por remoção do stride e dilatação 2 no `layer4`;
- ASPP e cabeça DeepLab autorais em `src/models.py`;
- argumentos `--decoder_type`, `--output_stride` e `--aspp_rates` no treino;
- persistência e reconstrução estrita da arquitetura nos checkpoints;
- cálculo teórico do campo receptivo em `src/utils.py`;
- testes estruturais em `tests/test_models.py`;
- grade reproduzível em `scripts/run_eixo1.sh` e consolidação em `scripts/summarize_eixo1.py`.

Os seis treinos devem usar os mesmos pesos de classe explícitos. Depois de os
pesos serem calculados uma única vez no split de treino, a grade é iniciada por:

```bash
PA1_CLASS_WEIGHTS="w_fundo,w_interior,w_fronteira" bash scripts/run_eixo1.sh
```

O script recusa sobrescrever qualquer execução existente e gera
`outputs/part3_eixo1/summary.json` com média e desvio-padrão amostral das seeds.
