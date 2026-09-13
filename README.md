# PA1 - Segmentação de Instâncias com Arquiteturas Densas

Programming Assignment 1 da disciplina de Aprendizado Profundo (FGV EMAp).  
Segmentação de instâncias em microscopia celular (DSB2018 / BBBC038v1) utilizando arquiteturas densas com representações e decodificadores autorais.

## Autores
- [Henrique Coelho Beltrão](https://github.com/riqueu)
- [Isaias Gouvêa Gonçalves](https://github.com/isaiasgoncalves)

---

## 1. Visão Geral e Resultados-Chave

O projeto desenvolve uma solução autoral de segmentação de instâncias baseada em **U-Net com encoder ResNet-34**, desacoplamento por **fronteiras de 1 px e decodificação Watershed (Trilha A)**, superando o baseline semântico e solucionando problemas práticos de mosaico, falhas morfológicas e robustez a escala:

| Etapa / Experimento | Abordagem Avaliada | Comparação / Linha de Base | mAP@[.50:.95] | Ganho / Variação | Erro de Contagem |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Parte 1: Baseline** | U-Net Binária + Componentes Conexos | Segmentação Semântica Pura | 0.4820 | — | 8.55 núcleos/img |
| **Parte 2: Modelo Final** | U-Net 3-Canais + Watershed (Trilha A) | vs. Baseline Semântico (Parte 1) | **0.5157** | **+3.37 pp** | **7.40 núcleos/img** |
| *Caso Denso (369 núcleos)* | *Watershed Desacoplado* | *vs. Fusão por Componentes Conexos* | **0.3374** | **+19.31 pp** | **47** *(vs. 174)* |
| **Parte 3: Ablações** | U-Net com Skip Connections | vs. Gargalo Cego / ASPP | **0.4901** | **+27.29 pp** *(vs. Gargalo)* | **5.99 núcleos/img** |
| **Parte 4: Mosaico** | Costura de Instâncias (Union-Find) | vs. Tiling Ingênuo (sem fusão) | **0.4413** | **+5.15 pp** | **+4** *(vs. +33 duplicadas)* |
| **Parte 5: Correção** | Watershed Adaptativo (fechamento $r=4$) | vs. Watershed Padrão (hiper-fragmentado) | **0.2518** | **+17.78 pp** | **0** *(exato, vs. +52)* |
| **Parte 6: Estresse (0.5×)** | U-Net ResNet-34 (multiescala) | vs. DeepLabv3 ASPP (dilatação fixa) | **0.3446** | **+32.70 pp** *(ASPP = 0.0176)* | **11.12** *(vs. 37.40)* |

---

## 2. Configuração do Ambiente

Python 3.10+ em ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Dados (DSB2018 / BBBC038v1)

1. Baixe o `stage1_train.zip` (82.9 MB) e `metadata.xlsx` do [Broad Institute](https://bbbc.broadinstitute.org/BBBC038).
2. Extraia os dados mantendo a estrutura:
```bash
mkdir -p data/raw/stage1_train
unzip stage1_train.zip -d data/raw/stage1_train
mv metadata.xlsx data/raw/ && rm stage1_train.zip
```
*Os splits estratificados por modalidade estão salvos em `data/splits.json`.*

---

## 4. Execução Rápida

### 4.1. Comandos Principais (Edital)

- **Treinamento do Modelo Final Oficial (Parte 2 — Trilha A Watershed):**
  ```bash
  python train.py --dataset dsb2018 --epochs 15 --batch_size 8 --lr 1e-3 --out_channels 3 --loss weighted_ce_3c --class_weights auto --out runs/part2_watershed --checkpoint checkpoints/part2_watershed.pth
  ```

- **Avaliação Oficial em Lote (Hungarian Matching mAP@[.50:.95]):**
  ```bash
  python evaluate.py --checkpoint checkpoints/part2_watershed.pth --dataset dsb2018 --split val --matching hungarian --output-dir outputs/part2_eval
  ```

### 4.2. Comandos Adicionais para Reprodução

```bash
# Parte 0 (Sanidade Sintética, <5 min, IoU > 0.99)
python train.py --dataset synthetic --epochs 10 --num_samples 500 --batch_size 16 --out runs/part0_synthetic --checkpoint checkpoints/part0_synthetic.pth

# Parte 1 (Baseline Semântico Binário)
python train.py --dataset dsb2018 --epochs 15 --batch_size 8 --lr 1e-3 --out runs/part1_baseline --checkpoint checkpoints/part1_baseline.pth

# Parte 3 (Ablações dos Eixos 1 e 2)
bash scripts/run_eixo1.sh && python scripts/summarize_eixo1.py
bash scripts/run_eixo2.sh && python scripts/summarize_eixo2.py

# Parte 4 (Mosaico & Costura com Union-Find)
python scripts/run_mosaic_demo.py --checkpoint checkpoints/part2_watershed.pth

# Parte 5 (Galeria de Falhas, Campo Receptivo e Correção Adaptativa)
python scripts/run_failure_gallery.py

# Parte 6 (Teste de Estresse de Escala 0.5x e 2.0x)
python scripts/run_scale_stress.py --unet-checkpoint checkpoints/part2_watershed.pth --aspp-checkpoint checkpoints/part3_eixo1/deeplab_aspp_seed42.pth
```

---

## 5. Inferência Arbitrária e Checkpoints

- **Notebook de Inferência:** [`notebooks/inferencia.ipynb`](notebooks/inferencia.ipynb) atua como vitrine técnica do projeto. Contém a função `predict_instances(image_path, model)` que recebe uma imagem qualquer e retorna a máscara colorida e a contagem sem retreinar.
- **Checkpoints Oficiais:** Os pesos estão disponíveis para download na [Release v1.0.0](https://github.com/riqueu/dl-pa1-2026/releases/tag/v1.0.0):
  - [Download `part2_watershed.pth`](https://github.com/riqueu/dl-pa1-2026/releases/download/v1.0.0/part2_watershed.pth): Modelo final oficial ($mAP = 0.5157$).
  - [Download `part1_baseline.pth`](https://github.com/riqueu/dl-pa1-2026/releases/download/v1.0.0/part1_baseline.pth): Baseline semântico binário ($mAP = 0.4820$).
  - [Download `part0_synthetic.pth`](https://github.com/riqueu/dl-pa1-2026/releases/download/v1.0.0/part0_synthetic.pth): Teste de sanidade sintético ($IoU > 0.99$).

Download rápido via terminal:
```bash
wget -P checkpoints/ https://github.com/riqueu/dl-pa1-2026/releases/download/v1.0.0/part2_watershed.pth
```

---

## 6. Registro de IA e Planejamento

- [AI_LOG.md](AI_LOG.md): Registro sucinto de uso de IA (5 episódios temáticos).
- [docs/](docs/): Roteiros de divisão de tarefas e contratos de interface entre a dupla.
