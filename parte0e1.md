Abaixo está o **Plano Definitivo de Ataque para a Fase Inicial (Partes 0 e 1)**, que estabelece os contratos e os arquivos exatos a serem implementados antes de entrar na escolha de trilhas da Parte 2.

---

### 1. Contratos Estritos de Interface

Fixem esses formatos antes de escrever a primeira linha de código:

* **Imagens (`image`):** Tensor float32 `(B, 3, H, W)` normalizado em $[0.0, 1.0]$. (Tanto sintético quanto real).
* **Máscara Semântica Binária (`mask_semantic`):** Tensor float32 `(B, 1, H, W)` com valores $\{0.0, 1.0\}$.
* **Máscara de Instâncias (`mask_instance`):** Tensor int64 `(B, H, W)` com $0 = \text{fundo}$ e $1, 2, \dots, K = \text{IDs de instâncias individuais}$.
* **Saída da Rede Baseline (`logits`):** Tensor float32 `(B, 1, H, W)`.
* **Saída do Pós-processador Ingênuo (`postprocess.py`):** Array NumPy `(H, W)` com rótulos inteiros $\{0, 1, \dots, K\}$.
* **Entrada e Saída da Métrica (`metrics.py`):**
```python
def evaluate_instances(pred_mask: np.ndarray, gt_mask: np.ndarray, iou_thresholds=None) -> dict:
    # pred_mask: (H, W) int
    # gt_mask:   (H, W) int
    # Retorna: {'mAP': float, 'AP_per_iou': dict, 'count_error': int}

```



---

### 2. Divisão das Tarefas Iniciais

#### Membro 1 (Henrique) — Pilar de Dados & Métricas

* **Branch:** `feature/dataset-and-metrics`
* **Módulos:** `src/dataset.py` e `src/metrics.py`
* **Escopo imediato:**
1. **Gerador Sintético (Parte 0):** Criar `SyntheticDataset(Dataset)` que desenha proceduralmente de 5 a 20 elipses sobrepostas em canvas $128 \times 128$ com ruído gaussiano/sal e pimenta e contraste variável, devolvendo `image` (3 canais), `mask_semantic` e `mask_instance`.


2. **Matching & mAP (Partes 0 e 1):** Criar em `src/metrics.py` o algoritmo de matching do zero (Hungarian via `scipy.optimize.linear_sum_assignment` ou Greedy por IoU decrescente) varrendo IoU de 0.50 a 0.95 com passo 0.05, calculando TP, FP, FN, mAP e $\vert{}N_{pred} - N_{gt}\vert{}$ (erro de contagem).


3. **DSB2018 & Split Estratificado (Parte 1):** Criar o parser do `data/raw/stage1_train/` e a função de split estratificado por `image_group` do `metadata.xlsx`, salvando os splits em `data/splits.json`.





#### Membro 2 (Isaias) — Pilar de Modelagem, Perdas & Pipeline

* **Branch:** `feature/models-and-training`
* **Módulos:** `src/models.py`, `src/losses.py`, `src/postprocess.py`, `train.py`
* **Escopo imediato:**
1. **Arquitetura Baseline (Partes 0 e 1):** Implementar em `src/models.py` uma U-Net clássica com encoder pré-treinado (ex: ResNet18/34 de `torchvision.models` extraindo 4 escalas de features) e um decoder próprio com upsampling/transpose conv + skip connections, saindo 1 canal (logits).


2. **Funções de Perda:** Implementar em `src/losses.py` a perda combinada BCE com pesos/logits + Dice Loss (e já deixar engatilhada a Balanced Focal Loss parametrizada por $\alpha$ e $\gamma$ para as ablações).


3. **Pós-processamento Ingênuo (Parte 1):** Implementar em `src/postprocess.py` a função `naive_connected_components(logits, threshold=0.5)` usando `scipy.ndimage.label`.


4. **CLI de Treinamento (`train.py`):** Criar o loop de treino com argumentos de linha de comando (`--dataset synthetic|dsb2018`, `--epochs`, `--lr`, `--batch_size`, `--seed`).



---

### 3. Cronograma de Execução: Marco a Marco

| Etapa | Responsável | Entregável / Ação | Critério de Sucesso |
| --- | --- | --- | --- |
| **Passo 1** | Ambos | Criação das branches e implementação isolada dos contratos. | Zero conflitos em commits. |
| **Passo 2** | Ambos | Merge das duas branches na `main` via PRs. | `src/` unificado e funcional. |
| **Passo 3** | Juntos | **Execução da Parte 0**: Rodar `python train.py --dataset synthetic --epochs 10`. | Treino convergiu em **< 5 minutos** na GPU, validado com métricas do `metrics.py`. |
| **Passo 4** | Juntos | **Execução da Parte 1**: Rodar treino na base real `dsb2018` usando o split estratificado.  | Obtenção de IoU, Dice, mAP@[.50:.95] e contagem pelo método de componentes conexos.|
| **Passo 5** | Juntos | Gerar o gráfico de fracasso da Parte 1: mAP (ou erro de contagem) vs. densidade de núcleos por imagem. | Tendência de queda de performance em áreas densas evidenciada. |

Concluídos os passos 3 a 5, toda a fundação obrigatória (Partes 0 e 1) estará finalizada e validada. A partir desse ponto, o projeto bifurca para a escolha da trilha de representação de instâncias (Trilha A com Watershed, Trilha B com Embeddings ou Trilha C com Centros/Offsets).
