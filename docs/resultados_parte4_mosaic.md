# Resultados da Parte 4 — Mosaico Grande, Tiling Ingênuo e Diagnóstico de Falha
**Responsável:** Membro A (Henrique)  
**Branch de Desenvolvimento:** `feature/part4-mosaic-pipeline`  
**Data:** 11/09/2026  

---

## 1. Contexto e Motivação Teórica

Conforme apresentado no slide 83 da aula e estipulado no edital do PA1:
> *"Imagens grandes são processadas em tiles, com patches sobrepostos, considerando a parte interna e fazendo a média dos resultados. Isso funciona para segmentação semântica. Para instâncias isso não funciona muito bem (...):*
> 1. Montem uma imagem grande (mosaico de várias imagens do dataset).
> 2. Rodem a inferência em tiles com sobreposição.
> 3. Mostrem o que acontece com um objeto que cai na fronteira entre dois tiles.
> 4. Proponham e implementem uma correção: fusão de instâncias entre tiles. Meçam o mAP antes e depois da correção."

### O Colapso Conceitual da Inferência Ingênua em Instâncias
Em segmentação semântica, uma média ponderada suave sobre probabilidades de patches sobrepostos resolve a costura porque cada pixel possui apenas uma categoria. Em **segmentação de instâncias**, no entanto, cada objeto é uma entidade discreta com identificador único (ID):
1. **Fatiamento Espúrio na Fronteira:** Se o recorte central (*inner crop* / partição de Voronoi) for aplicado na junção dos tiles (prática padrão da semântica), qualquer núcleo que cruzar a linha de corte é fatiado em dois pedaços desconexos: a metade esquerda ganha um ID no Tile A e a metade direita ganha outro ID no Tile B.
2. **Duplicação de Objetos na Sobreposição:** Se cada tile for carimbado diretamente com novos identificadores sem comunicação entre patches, todo núcleo contido na faixa de sobreposição de $50\%$ é predito múltiplas vezes por tiles vizinhos.
3. **Colapso de mAP no Matching Húngaro:** Os fragmentos menores não atingem o limiar de $\text{IoU} \ge 0.50$ com o Ground Truth unificado (sendo computados como Falsos Negativos e Falsos Positivos simultaneamente), e a contagem total de objetos inflaciona severamente.

---

## 2. Pipeline Implementado (`src/mosaic.py` e `scripts/run_mosaic_demo.py`)

1. **Montador do Mosaico (`create_mosaic_sample`):**
   - Agrupa 4 imagens de $256 \times 256$ do DSB2018 em uma grade $2 \times 2$, gerando um canvas de $512 \times 512 \times 3$.
   - Unifica as anotações Ground Truth com identificadores globais estritamente consecutivos: $1 \dots K_{\text{total}}$, sem colisões de IDs.
2. **Extrator Contínuo de Alta Resolução (`create_mosaic_from_large_image`):**
   - Permite avaliar lâminas biológicas contínuas de alta resolução do dataset (ex.: ISBI TissueBW de $1024 \times 1024$ ou lâminas de $696 \times 520$), onde o tecido é contínuo e dezenas de células cruzam naturalmente as coordenadas das janelas deslizantes.
3. **Gerador de Janelas Deslizantes (`extract_tiles`):**
   - Gera patches de $256 \times 256$ com stride de 128 pixels (sobreposição de $50\%$). Em um canvas de $512 \times 512$, produz uma grade uniforme de $3 \times 3 = 9$ tiles.
4. **Inferência por Tile e Tiling Ingênuo (`predict_tiled_naive`):**
   - Executa a inferência individual do modelo Watershed oficial (`checkpoints/part2_watershed.pth`) em cada tile.
   - Avalia os dois modos de reconstrução ingênua:
     - `center_crop`: Particionamento pelo centro de cada tile (prática do slide 83).
     - `direct_stamp`: Estampa direta de todas as detecções com novos IDs sequenciais.
5. **Detector de Quebra Geométrica (`detect_split_nuclei`):**
   - Cruza a geometria dos núcleos reais do Ground Truth com as coordenadas das linhas de corte das janelas deslizantes ($x, y \in \{128, 256, 384\}$), identificando quantos objetos foram interceptados e em quantos fragmentos preditos foram divididos.

---

## 3. Resultados Quantitativos

A tabela abaixo compara o desempenho do Ground Truth contra os modos de Tiling Ingênuo tanto no **Mosaico em Grade 2x2** (4 imagens densas de validação, 310 núcleos reais) quanto na **Lâmina Biológica Contínua** (`0ea22171...`, $512 \times 512$, 292 núcleos reais).

| Cenário de Teste | Modo de Reconstrução | N° Instâncias Preditas | mAP@[.50:.95] | AP@0.50 | AP@0.75 | Erro de Contagem ($\Delta N$) | Núcleos na Borda | Núcleos Fatiados ($\ge 2$ IDs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mosaico Grade 2x2** (310 GT) | *Single-Image Baseline* | — | $\approx 0.5157$ | $\approx 0.7740$ | $\approx 0.5520$ | $\approx 5.8$ | — | — |
| **Mosaico Grade 2x2** (310 GT) | **Tiling Ingênuo (Center-Crop)** | **343** | **0.3898** | **0.6744** | **0.3983** | **+33** | 37 (11.9%) | **10** |
| **Mosaico Grade 2x2** (310 GT) | **Tiling Ingênuo (Direct-Stamp)** | **408** | **0.3549** | **0.5920** | **0.3888** | **+98** | 37 (11.9%) | N/A (duplicados) |
| **Lâmina Contínua** (292 GT) | **Tiling Ingênuo (Center-Crop)** | **387** | **0.2079** | **0.4478** | **0.1871** | **+95** | 64 (21.9%) | **35** |

### Principais Constatações:
1. **Queda Drástica de mAP:** No mosaico em grade, o mAP cai de $0.5157$ para $0.3898$ (perda de **12.6 pontos percentuais**). Na lâmina contínua, onde $21.9\%$ dos núcleos cruzam as linhas de corte, o mAP despenca para **$0.2079$** (perda de mais de **30 pontos percentuais**).
2. **Inflação Espúria de Contagem:**
   - No modo `center_crop`, a contagem salta de 310 para 343 (+33) no mosaico em grade e de 292 para 387 (+95) na lâmina contínua devido aos fragmentos fatiados que ganham identificadores independentes.
   - No modo `direct_stamp`, a falta de fusão nas zonas de sobreposição de $50\%$ gera 408 instâncias (+98 objetos fantasmas), penalizando o matching húngaro com dezenas de Falsos Positivos redundantes.
3. **Fatiamento Confirmado:** 10 núcleos no mosaico em grade e 35 núcleos na lâmina contínua foram comprovadamente fatiados pela borda dos tiles em dois ou mais fragmentos com IDs distintos.

---

## 4. Evidências Visuais da Falha

Os gráficos gerados automaticamente encontram-se persistidos em `outputs/part4_mosaic/`:

1. **Mosaico em Grade 2x2 (`outputs/part4_mosaic/tiling_naive_failure.png`):**
   - Mostra o painel comparativo em 4 vistas: Mosaico RGB com a grade de janelas deslizantes $\to$ Ground Truth biológico contínuo $\to$ Predição do Tiling Ingênuo $\to$ Zoom in de alta resolução na fronteira.
   - No painel de zoom, a linha horizontal vermelha pontilhada ($y=128$) corta o núcleo celular exatamente ao meio: a metade superior é pintada de azul (ID $A$) e a metade inferior de vermelho (ID $B$).
2. **Lâmina Biológica Contínua (`outputs/part4_mosaic/tiling_naive_failure_continuous.png`):**
   - Evidencia a rede de costura ortogonal ($x, y \in \{128, 256, 384\}$). No painel de zoom, a linha vertical vermelha corta uma célula oblonga em um fragmento roxo (esquerda) e outro laranja (direita).

---

## 5. Contrato de Entrega para o Membro B (Isaias)

O pipeline do Membro A foi construído de forma estritamente desacoplada para alimentar o algoritmo de fusão do Membro B (`src/stitching.py`):

```python
from src.mosaic import create_mosaic_sample, predict_tiled_naive
from src.stitching import stitch_tiles_with_fusion

# 1. Obtenção das estruturas intermediárias desacopladas
res = predict_tiled_naive(mosaic_img, model, tile_size=256, stride=128)
tiles_preds = res["tiles_preds"]  # Lista de 9 arrays (256, 256) com instâncias locais
tile_boxes = res["tile_boxes"]    # Lista de 9 tuplas (ymin, xmin, ymax, xmax)

# 2. Execução da fusão de instâncias (Escopo do Membro B)
stitched_mask = stitch_tiles_with_fusion(
    tiles_preds=tiles_preds,
    tile_boxes=tile_boxes,
    full_shape=(512, 512),
    iou_overlap_threshold=0.20,
)
```

**Meta para a Fusão (Membro B):**  
Restabelecer os núcleos fatiados, eliminar duplicações na faixa de overlap e recuperar o mAP de volta para a faixa de $\ge 0.50$, reduzindo o erro de contagem de $+33$ para próximo de zero.
