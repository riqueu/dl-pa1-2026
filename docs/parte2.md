# Plano de Implementação da Parte 2 (Trilha A: Fronteiras e Watershed)

Este documento estabelece os contratos técnicos, a divisão de tarefas em paralelo para a dupla e o cronograma de execução da **Parte 2 — Cabeça de Instâncias** com a **Trilha A (Fronteiras e Watershed)**, dando continuidade aos resultados das Partes 0 e 1.

---

### 1. Contratos Estritos de Interface da Trilha A

Para garantir compatibilidade imediata entre o pipeline de dados e o modelo, os formatos de tensores e saídas são estritamente padronizados:

#### 1.1. Alvo de Treinamento (`mask_three_class`)
* **Tensor:** `torch.int64` de shape `(B, H, W)` ou `(H, W)` com valores no conjunto `{0, 1, 2}`:
  * **Classe 0 (Fundo):** Pixels fora de qualquer núcleo celular.
  * **Classe 1 (Interior):** Núcleo erodido morfologicamente por um elemento estruturante (disco/quadrado de raio $r=1$ ou $2$ pixels). Garante que dois núcleos adjacentes fiquem separados espacialmente.
  * **Classe 2 (Fronteira):** Borda e região de contato entre núcleos ($M_{\text{semântica}} \setminus M_{\text{interior}}$), além de zonas de contato entre instâncias vizinhas dilatadas.

#### 1.2. Saída da Rede (`logits` e `probs`)
* **`logits`:** Tensor `torch.float32` de shape `(B, 3, H, W)`.
* **`probabilities`:** Tensor `torch.float32` de shape `(B, 3, H, W)` normalizado via `torch.softmax(logits, dim=1)`.
  * Canal 0: $P(\text{fundo})$
  * Canal 1: $P(\text{interior})$
  * Canal 2: $P(\text{fronteira})$

#### 1.3. Pós-processamento Watershed (`src/postprocess.py`)
* **Assinatura:**
  ```python
  def watershed_instance_segmentation(
      prob: np.ndarray,
      interior_threshold: float = 0.50,
      foreground_threshold: float = 0.50,
      min_area: int = 10,
  ) -> Tuple[np.ndarray, int]:
      """
      Args:
          prob: Array numpy (3, H, W) com as probabilidades [fundo, interior, fronteira].
          interior_threshold: Limiar para binarizar as sementes de interior.
          foreground_threshold: Limiar para a máscara total de núcleos (interior + fronteira).
          min_area: Filtro de remoção de ruídos espúrios.
      Returns:
          instance_mask: Array numpy (H, W) int64 com 0 = fundo e 1..K = IDs de instâncias.
          num_instances: Contagem total K de núcleos detectados.
      """
  ```

#### 1.4. Métrica de Avaliação (`src/metrics.py`)
* Mantém a mesma interface validada da Parte 1:
  ```python
  evaluate_instances(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict
  # Retorna: {'mAP': float, 'AP_per_iou': dict, 'count_error': int}
  ```

---

### 2. Divisão de Tarefas em Paralelo

A divisão entre os membros mantém o isolamento estrito de arquivos para evitar qualquer conflito de merge:

```
                  ┌────────────────────────────────────────┐
                  │              main (Partes 0 e 1)       │
                  └───────────────────┬────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   feature/part2-data-targets              feature/part2-watershed-head
   (Membro 1 - Henrique)                   (Membro 2 - Isaias)
   • src/dataset.py                        • src/models.py (out_channels=3)
   • Geração de 3 classes (Erosão)         • src/losses.py (Weighted CE/Focal)
   • Cálculo de pesos de classes           • src/postprocess.py (Watershed)
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                        Integração & Treino Conjunto
                        • train.py & evaluate.py
                        • checkpoints/best_model.pth
                        • notebooks/inferencia.ipynb
```

#### Membro 1 (Henrique) — Pilar de Dados & Alvos de 3 Classes
* **Branch:** `feature/part2-data-targets`
* **Arquivos:** `src/dataset.py`, testes locais em `scratch/`
* **Escopo:**
  1. **Algoritmo de 3 Classes:** Implementar função `generate_three_class_mask(instance_mask, radius=1)` usando `scipy.ndimage.binary_erosion`:
     - Para cada núcleo individual $k \in \{1, \dots, K\}$, erodir a máscara binária individual por raio $r=1$ (ou $r=2$ para núcleos grandes).
     - Atribuir classe 1 para os núcleos erodidos.
     - Atribuir classe 2 (fronteira) para a diferença entre a máscara semântica total e os interiores erodidos.
  2. **Atualização do Dataset:** Adicionar flag `three_class=True` em `DSB2018Dataset` e `SyntheticDataset`, retornando a chave `'mask_target'` no dicionário de cada amostra.
  3. **Cálculo de Frequências de Classes:** Calcular a proporção média de pixels de fundo, interior e fronteira no conjunto de treino para calibrar os pesos da função de perda (espera-se algo próximo a 80% fundo, 15% interior, 5% fronteira).

#### Membro 2 (Isaias) — Pilar de Modelagem, Perda Ponderada & Watershed
* **Branch:** `feature/part2-watershed-head`
* **Arquivos:** `src/models.py`, `src/losses.py`, `src/postprocess.py`
* **Escopo:**
  1. **Modelo de 3 Classes (`src/models.py`):**
     - Garantir suporte limpo a `build_model(encoder='resnet34', out_channels=3)`.
     - Preservar compatibilidade total com o modelo binário (`out_channels=1`).
  2. **Função de Perda Ponderada (`src/losses.py`):**
     - Implementar `WeightedCrossEntropyLoss(weights=[w_bg, w_interior, w_boundary])`, dando peso ampliado (ex.: peso 5.0 a 10.0) para a classe fronteira.
     - Implementar `MulticlassFocalLoss(gamma=2.0, alpha=weights)` como alternativa de ablação.
     - Adicionar suporte a `loss_type='weighted_ce_3c'` em `build_loss()`.
  3. **Pós-processador Watershed (`src/postprocess.py`):**
     - Implementar `watershed_instance_segmentation(prob)`:
       a) Identificar sementes: `ndimage.label(prob[1] > interior_thresh)`.
       b) Construir superfície de inundação (energia topográfica): probabilidade de fronteira `prob[2]` ou inverso da probabilidade de interior `1.0 - prob[1]`.
       c) Máscara de barreira: `prob[0] < 0.5` (restringe o watershed a não invadir o fundo).
       d) Executar `skimage.segmentation.watershed` ou algoritmo de bacia hidrográfica equivalente.

---

### 3. Cronograma Marco a Marco

| Marco | Responsável | Ação / Entregável | Critério de Sucesso |
| :--- | :---: | :--- | :--- |
| **Passo 1** | Membro 1 (Henrique) | Implementar geração das 3 classes em `src/dataset.py`. | Amostras densas mostram núcleos adjacentes com interiores desacoplados e bordas bem marcadas. |
| **Passo 2** | Membro 2 (Isaias) | Implementar `out_channels=3`, perda ponderada e `watershed` em `postprocess.py`. | Teste sintético do pós-processamento separa perfeitamente marcadores artificiais. |
| **Passo 3** | Ambos | Merge das duas branches na `main` via Pull Request. | Zero conflitos de código; imports e pipeline 100% integrados. |
| **Passo 4** | Juntos | Treinamento da U-Net Trilha A no DSB2018 (15 épocas):<br>`python train.py --dataset dsb2018 --model_type watershed --epochs 15`. | Perda de fronteira converge; checkpoint salvo em `checkpoints/part2_watershed.pth`. |
| **Passo 5** | Juntos | Avaliação em lote via `evaluate.py`: mAP@[.50:.95] e erro de contagem. | **mAP@[.50:.95] supera expressivamente o baseline (meta: > 0.55)** e o erro de contagem em imagens densas cai drasticamente. |
| **Passo 6** | Juntos | Atualização do `notebooks/inferencia.ipynb` e `AI_LOG.md`. | Comparação visual tripla (Original vs Baseline vs Watershed) renderizada no caso crítico de 369 núcleos. |
