# Plano de Implementação da Parte 4 — Inferência em Mosaico (Tiling & Stitching)

Este documento estabelece o plano técnico, os contratos de código e a divisão de tarefas em paralelo para a execução da **Parte 4 — Inferência em Mosaico** do PA1 (Deep Learning — FGV EMAp).

Conforme estipulado no edital:
> *"O último slide prático da aula (83) descreve a prática padrão: imagens grandes são processadas em tiles, com patches sobrepostos, considerando a parte interna e fazendo a média dos resultados. Isso funciona para segmentação semântica. Para instâncias isso não funciona muito bem (...):*
> 1. Montem uma imagem grande (mosaico de várias imagens do dataset).
> 2. Rodem a inferência em tiles com sobreposição.
> 3. Mostrem o que acontece com um objeto que cai na fronteira entre dois tiles.
> 4. Proponham e implementem uma correção: fusão de instâncias entre tiles. Meçam o mAP antes e depois da correção."*

---

## 1. O Problema Conceitual: Por que Mosaicos Quebram Instâncias?

Em **segmentação semântica**, cada pixel recebe apenas uma classe categórica. Se dois tiles sobrepostos predizem probabilidades para o mesmo pixel, fazer a média ponderada (com pesos maiores no centro do patch via janela Gaussiana/Hann) produz uma transição suave e contínua.

Em **segmentação de instâncias**, cada objeto possui um **identificador discreto e único (ID)**:
1. **Fatiamento Espúrio:** Um núcleo celular que cruza a linha de corte entre o Tile A e o Tile B será detectado duas vezes: a metade esquerda ganha um ID no Tile A e a metade direita ganha outro ID no Tile B.
2. **Duplicação e Deformação:** O objeto é fatiado em dois fragmentos menores, inflando a contagem total e reduzindo drasticamente o IoU contra o Ground Truth original.
3. **Colapso de mAP:** O Hungarian matching penaliza tanto os fragmentos como falsos positivos parciais quanto a falta do objeto completo como falso negativo.

---

## 2. Solução Proposta: Costura com Fusão de Instâncias (Instance Stitching)

Para corrigir a falha, implementaremos um algoritmo autoral de costura baseado em grafo de adjacência na faixa de sobreposição (*overlap strip*):

```
       Tile A (Esquerda)                     Tile B (Direita)
┌───────────────────────┬───────────────┐
│                       │   Sobreposição│                       │
│      Instância k_A    │      (Overlap)│     Instância k_B     │
│       (Metade 1)      │  ┌─────────┐  │      (Metade 2)       │
│                       │  │k_A ∩ k_B│  │                       │
│                       │  └─────────┘  │                       │
└───────────────────────┴───────────────┴───────────────────────┘
                                ▲
                IoU_overlap(k_A, k_B) >= τ_fuse
                                │
                    FUSÃO EM UM ÚNICO OBJETO
```

1. **Extração de Pares Candidatos:** Para cada par de tiles adjacentes, identificamos as instâncias que intersectam a região espacial compartilhada de overlap.
2. **Critério de Compatibilidade:** Calculamos o IoU restrito à faixa de sobreposição:
   $$\text{IoU}_{\text{overlap}}(i, j) = \frac{|\mathcal{M}_{A, i} \cap \mathcal{M}_{B, j}|}{|\mathcal{M}_{A, i} \cup \mathcal{M}_{B, j}|}$$
3. **Fusão Conexa (Disjoint Set / Union-Find):** Se $\text{IoU}_{\text{overlap}} \ge \tau_{\text{fuse}}$ (ex.: $\tau = 0.20$), unificamos os identificadores $i$ e $j$ na mesma componente conexa de instâncias globais.
4. **Resolução de Conflitos:** Pixels na faixa de sobreposição onde não há fusão são arbitrados pela distância euclidiana ao centro de cada tile (confiança do campo receptivo).

---

## 3. Divisão de Tarefas da Dupla

```
                               ┌────────────────────────────────────────┐
                               │           main (Partes 2 e 3)          │
                               └───────────────────┬────────────────────┘
                                                   │
                      ┌────────────────────────────┴────────────────────────────┐
                      ▼                                                         ▼
        feature/part4-mosaic-pipeline                              feature/part4-instance-stitching
        (Membro A - Henrique)                                      (Membro B - Isaias)
        • src/mosaic.py                                            • src/stitching.py
        • Montador de Mosaico (2x2 de 512x512)                     • Algoritmo de Fusão de Instâncias (IoU overlap)
        • Mapeamento de GT global contínuo                         • Resolução de bordas e re-indexação
        • Tiling ingênuo e evidência da falha                      • Avaliação mAP antes vs. depois da fusão
                      │                                                         │
                      └────────────────────────────┬────────────────────────────┘
                                                   ▼
                                       Integração & Demonstração
                                       • notebooks/inferencia.ipynb (Seção 4)
                                       • Painel visual comparativo
```

### 3.1. Membro A (Henrique) — Mosaico Grande & Tiling Ingênuo
* **Branch:** `feature/part4-mosaic-pipeline`
* **Arquivo:** `src/mosaic.py`
* **Escopo:**
  1. **Montador do Mosaico (`build_mosaic`):**
     - Seleciona 4 imagens do DSB2018 (ex.: $256 \times 256$) e monta um mosaico $2 \times 2$ de $512 \times 512$ pixels (ou costura imagens com núcleos forçados na fronteira).
     - Concatena as máscaras de instâncias ground truth, preservando identificadores únicos globais ($1 \dots K_{\text{total}}$).
  2. **Pipeline de Tiling Ingênuo (`predict_tiled_naive`):**
     - Divide a imagem grande em janelas de $256 \times 256$ com sobreposição de $50\%$ (stride de 128 px).
     - Executa a inferência em cada tile separadamente com o modelo da Parte 2 (Watershed).
     - Insere as instâncias de volta na imagem grande atribuindo novos IDs a cada tile sem comunicação entre fronteiras.
  3. **Visualização da Falha:**
     - Gerar figura destacando com zoom os núcleos cortados pela fronteira do patch que foram duplicados.

---

### 3.2. Membro B (Isaias) — Algoritmo de Fusão & Avaliação de Ganho
* **Branch:** `feature/part4-instance-stitching`
* **Arquivo:** `src/stitching.py`
* **Escopo:**
  1. **Algoritmo de Fusão (`stitch_instances`):**
     - Recebe a lista de predições de instâncias por tile e suas coordenadas espaciais $(x, y, w, h)$.
     - Varre as faixas de interseção horizontal e vertical entre patches vizinhos.
     - Monta matriz de adjacência de sobreposição e aplica Union-Find para fundir IDs correspondentes.
     - Gera a máscara final unificada $(H_{\text{mosaico}}, W_{\text{mosaico}})$ com rótulos consecutivos.
  2. **Avaliação Comparativa Formal:**
     - Executa `evaluate_instances` sobre o Ground Truth global do mosaico:
       * **Antes da Correção (Tiling Ingênuo):** Medir $\text{mAP}@[.50:.95]$ e contagem inflada.
       * **Depois da Correção (Com Fusão):** Medir $\text{mAP}@[.50:.95]$ recuperado e contagem real.
  3. **Visualização Antes vs. Depois:**
     - Painel visual lado a lado: Imagem $\to$ GT $\to$ Tiling Ingênuo (duplicado) $\to$ Após Fusão (unificado).

---

## 4. Estrutura de Arquivos e Interfaces

```python
# src/mosaic.py
def create_mosaic_sample(image_ids: List[str], target_shape=(512, 512)) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna a imagem mosaico (H, W, 3) e a máscara de instâncias combinada (H, W)."""

def extract_tiles(image: np.ndarray, tile_size=256, stride=128) -> List[dict]:
    """Gera recortes sobrepostos com coordenadas (ymin, xmin, ymax, xmax)."""

# src/stitching.py
def stitch_tiles_with_fusion(
    tiles_preds: List[np.ndarray],
    tile_boxes: List[Tuple[int, int, int, int]],
    full_shape: Tuple[int, int],
    iou_overlap_threshold: float = 0.20,
) -> np.ndarray:
    """Funde as instâncias sobrepostas e devolve a máscara global contínua (H, W)."""
```

---

## 5. Cronograma Marco a Marco para as Partes 3 e 4

| Marco | Responsável | Parte | Ação / Entregável | Critério de Sucesso |
| :--- | :---: | :---: | :--- | :--- |
| **P3.1** | Isaias | Parte 3 | Implementar ASPP em `src/models.py` e rodar 6 treinos do Eixo 1. | Mapeamento multiescala funcional; média $\pm$ desvio tabulada. |
| **P3.2** | Henrique | Parte 3 | Automação da grade de $\gamma$ no Eixo 2 e rodar 10 treinos. | Efeito do desbalanceamento da fronteira ($2.8\%$) quantificado. |
| **P3.3** | Ambos | Parte 3 | Consolidação da Parte 3 no notebook e merge na `main`. | Tabelas comparativas de ablações preenchidas e discutidas. |
| **P4.1** | Henrique | Parte 4 | Criar `src/mosaic.py`, montar mosaico $512\times 512$ e tiling ingênuo. | Demonstração clara da fragmentação e duplicação de núcleos. |
| **P4.2** | Isaias | Parte 4 | Criar `src/stitching.py` com algoritmo de união na sobreposição. | Fusão restabelece núcleos cortados e reduz erro de contagem. |
| **P4.3** | Ambos | Parte 4 | Avaliação comparativa mAP antes/depois da fusão e integração final. | Ganho expressivo de mAP reportado após a costura de instâncias. |

---

## 6. Estado da Implementação da Fusão

A branch `feature/part4-instance-stitching` deriva diretamente de
`feature/part3-eixo1-resolution`, preservando a hierarquia acordada para o
desenvolvimento.

O módulo `src/stitching.py` já oferece:

- validação do contrato de máscaras locais e caixas globais semiabertas;
- atribuição de identidades provisórias únicas por `(tile, id_local)`;
- matching guloso um-para-um por IoU restrito à faixa de sobreposição;
- Union-Find com compressão de caminho e união por tamanho;
- resolução de conflitos pela proximidade normalizada ao centro do tile;
- renumeração final dos IDs para `1..K`;
- composição sem fusão para medir formalmente o resultado antes da correção.

Os testes em `tests/test_stitching.py` cobrem fusão simples, não fusão de
objetos distintos, união transitiva em três tiles, tiles vazios, IDs não
consecutivos, validação de entradas e ganho de mAP no caso sintético. A
integração com `src/mosaic.py` dependerá apenas deste contrato:

```python
tiles_preds: list[np.ndarray]                 # máscaras (h, w), IDs locais
tile_boxes: list[tuple[int, int, int, int]]  # ymin, xmin, ymax, xmax
full_shape: tuple[int, int]                  # altura, largura
```
