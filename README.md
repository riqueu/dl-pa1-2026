# PA1 - Segmentação de Instâncias com Arquiteturas Densas

Repositório para o Programming Assignment 1 (PA1) da disciplina de Aprendizado Profundo (FGV EMAp).

## Dupla
- [Henrique Coelho Beltrão](https://github.com/riqueu)
- [Isaias Gouvêa Gonçalves](https://github.com/isaiasgoncalves)

---

## 1. Visão Geral do Projeto

Este projeto implementa e avalia métodos autorais de segmentação de instâncias para microscopia celular baseados no dataset **Data Science Bowl 2018 (DSB2018)**. O fluxo é estruturado em:
1. **Parte 0 (Teste de Sanidade Sintético):** Validação da U-Net autoral em dados procedurais de elipses sobrepostas com convergência estrita ($IoU > 0.99$).
2. **Parte 1 (Baseline Semântico):** U-Net com encoder ResNet34 pré-treinado e decoder autoral, extraindo instâncias ingenuamente via componentes conexos e quantificando o colapso de mAP em função da densidade de núcleos.
3. **Parte 2 (Instance Head — Trilha A: Fronteiras e Watershed):** Segmentação multitarefa em 3 classes ($P(\text{fundo}), P(\text{interior}), P(\text{fronteira})$) com Cross-Entropy balanceada e decodificação topográfica por Watershed, superando o baseline em mAP (+3.37 p.p. global e +19.31 p.p. no caso denso).
4. **Parte 3 (Ablações):** Eixo 1 (Recuperação de Resolução: U-Net Skips vs. DeepLab ASPP vs. No-Skips) e Eixo 2 (Perdas e Desbalanceamento: CE Ponderada vs. Focal Loss $\gamma \in \{0, 1, 2, 5\}$).
5. **Parte 4 (Mosaico & Costura):** Inferência em janelas deslizantes (tiling) e algoritmo autoral de fusão de instâncias na faixa de sobreposição (*Instance Stitching* via Union-Find).

### Resumo Comparativo de Desempenho (Validação DSB2018, Hungarian Matching)

| Parte / Abordagem | Configuração | mAP@[.50:.95] | AP @ IoU 0.50 | Erro Médio Contagem | IoU Semântico |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Parte 1: Baseline Semântico** | U-Net ResNet34 (Componentes Conexos) | 0.4820 | 0.6611 | 8.55 núcleos/img | **0.8439** |
| **Parte 2: Trilha A Oficial** | U-Net ResNet34 (Fronteiras e Watershed) | **0.5157** | **0.7165** | **7.40 núcleos/img** | 0.8236 |
| *Caso Crítico Denso (369 núcleos)* | *Baseline Semântico (Fusão Severa)* | *0.1443* | *—* | *174 núcleos (subcontagem)* | *—* |
| *Caso Crítico Denso (369 núcleos)* | *Trilha A (Watershed Desacoplado)* | **0.3374** | *—* | **47 núcleos (-73% erro)** | *—* |
| **Parte 3: Eixo 1 (Resolução)** | U-Net Padrão (Skip Connections) | **0.4901 ± 0.0210** | **0.7098 ± 0.0076** | **5.99 ± 0.11** | **0.8125 ± 0.0105** |
| **Parte 3: Eixo 1 (Resolução)** | U-Net sem Skips (Gargalo Cego) | 0.2172 ± 0.0713 | 0.4354 ± 0.1190 | 19.54 ± 6.97 | 0.6407 ± 0.0784 |
| **Parte 3: Eixo 1 (Resolução)** | DeepLabv3 (Atrous OS16 + ASPP) | 0.1588 ± 0.0008 | 0.3541 ± 0.0029 | 25.34 ± 0.91 | 0.6002 ± 0.0037 |
| **Parte 3: Eixo 2 (Perdas)** | Cross-Entropy Ponderada ($\alpha$) | **0.5002 ± 0.0155** | **0.7141 ± 0.0024** | 7.25 ± 0.15 | **0.8134 ± 0.0102** |
| **Parte 3: Eixo 2 (Perdas)** | Focal Loss ($\gamma = 0$) | **0.5015 ± 0.0205** | **0.7140 ± 0.0096** | **7.07 ± 0.37** | 0.8083 ± 0.0139 |
| **Parte 3: Eixo 2 (Perdas)** | Focal Loss ($\gamma = 5$) | 0.3636 ± 0.0059 | 0.6329 ± 0.0188 | 10.32 ± 0.96 | 0.7533 ± 0.0060 |
| **Parte 4: Mosaico 2x2 (310 GT)** | Tiling Ingênuo (Center-Crop) | 0.3898 | 0.6744 | +33 núcleos | — |
| **Parte 4: Mosaico 2x2 (310 GT)** | Costura com Fusão (Union-Find, $\tau=0.20$) | **0.4413 (+5.15 pp)** | **0.7479 (+7.35 pp)** | **+4 núcleos** | — |

```bash
dl-pa1-2026
├── AI_LOG.md              # Registro conciso de assistência de IA
├── checkpoints/           # Pesos dos modelos treinados (.pth)
├── data/                  # DSB2018 stage1_train e splits estratificados
├── docs/                  # Guias de planejamento e divisão de tarefas da dupla
│   ├── parte0e1.md        # Planejamento das Partes 0 e 1
│   ├── parte2.md          # Planejamento da Trilha A (Watershed)
│   ├── parte3.md          # Planejamento das Ablações (Eixos 1 e 2)
│   ├── parte4.md          # Planejamento de Mosaico e Costura de Instâncias
│   ├── parte5.md          # Planejamento da Galeria de Falhas e Diagnósticos
│   └── parte6.md          # Planejamento do Teste de Estresse (Escala e ASPP)
├── evaluate.py            # Avaliação em lote no conjunto de teste/validação
├── LICENSE
├── notebooks/
│   ├── exploratory.ipynb  # Prototipação e inspeção de dados
│   └── inferencia.ipynb   # Vitrine técnica completa (Partes 0 a 4)
├── PA1.pdf
├── README.md
├── requirements.txt
├── scripts/               # Scripts de automação e reprodução
│   ├── run_eixo1.sh       # Execução da grade do Eixo 1
│   ├── run_eixo2.sh       # Execução da grade do Eixo 2
│   ├── run_mosaic_demo.py # Demonstração integrada de tiling e fusão
│   ├── summarize_eixo1.py # Consolidação de métricas do Eixo 1
│   └── summarize_eixo2.py # Consolidação de métricas do Eixo 2
├── src/
│   ├── dataset.py         # Datasets reais, gerador sintético e alvos 3-classes
│   ├── __init__.py
│   ├── losses.py          # CE balanceada, Focal Loss multiclasse e pesos
│   ├── metrics.py         # Matching Hungarian/Greedy e cálculo de mAP@[.50:.95]
│   ├── models.py          # U-Net autoral, ASPP e variantes DeepLab
│   ├── mosaic.py          # Montagem de mosaicos e janelas deslizantes
│   ├── postprocess.py     # Componentes conexos e decodificação Watershed
│   ├── stitching.py      # Fusão de instâncias via IoU e Union-Find
│   └── utils.py           # Colorização, overlay e campo receptivo teórico
├── tests/                 # Suíte de testes unitários (12 testes passando)
└── train.py               # Pipeline de treino configurável via argumentos CLI
```

---

## 2. Configuração do Ambiente

Recomenda-se o uso de ambiente virtual (venv ou conda) com Python 3.10+.

Criar e ativar o ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate
```

Instalar dependências:
```bash
pip install -r requirements.txt
```

---

## 3. Obtenção e Organização dos Dados

Para o desenvolvimento das tarefas, utilizamos a **Opção A** (*DSB2018 / BBBC038v1* - microscopia de núcleos celulares).

### 3.1. Arquivos Necessários
Acesse o portal da base em https://bbbc.broadinstitute.org/BBBC038 e faça o download de:
1. `stage1_train.zip` (82.9 MB): Conjunto que contém tanto as imagens de microscopia quanto as máscaras individuais de cada núcleo.
2. `metadata.xlsx` (20 KB): Tabela de metadados utilizada para orientar o split estratificado por modalidade de microscopia.

Nota: Os arquivos `stage1_test.zip` e `stage2_test_final.zip` não são necessários.

### 3.2. Estrutura de Diretórios
Descompacte os dados locais para manter a seguinte árvore dentro do projeto:

```bash
data/
├── raw/
│   ├── metadata.xlsx
│   └── stage1_train/
│       ├── <ImageId_1>/
│       │   ├── images/
│       │   │   └── <ImageId_1>.png
│       │   └── masks/
│       │       ├── <MaskHash_1>.png
│       │       ├── <MaskHash_2>.png
│       │       └── ...
│       └── <ImageId_N>/
└── synthetic/
```

### 3.3. Download e Extração via Terminal

Via Kaggle CLI:
```bash
kaggle competitions download -c data-science-bowl-2018 -f stage1_train.zip
mkdir -p data/raw/stage1_train
unzip stage1_train.zip -d data/raw/stage1_train
rm stage1_train.zip
```

Via download manual [Broad Institute](https://bbbc.broadinstitute.org/BBBC038):
```bash
mkdir -p data/raw/stage1_train
unzip stage1_train.zip -d data/raw/stage1_train
mv metadata.xlsx data/raw/
rm stage1_train.zip
```

### 3.4. Estratificação e Splits

Conforme estabelecido nas especificações do trabalho, a separação entre treino, validação e teste é realizada de forma estratificada considerando a modalidade e condições experimentais descritas no `metadata.xlsx`. O pipeline de dados salva os índices das partições em disco para garantir reproducibilidade exata de todas as curvas e métricas.

---

## 4. Execução Rápida
 
### 4.1. Teste de Sanidade Sintético (Parte 0)
Treinamento rápido na base sintética com 500 imagens procedurais:
```bash
python train.py --dataset synthetic --epochs 10 --num_samples 500 --batch_size 16 --out runs/part0_synthetic --checkpoint checkpoints/part0_synthetic.pth
```
 
### 4.2. Treinamento do Baseline Semântico (Parte 1)
Treinamento da U-Net (ResNet34) na base real DSB2018 com BCE balanceada + Soft Dice:
```bash
python train.py --dataset dsb2018 --epochs 15 --batch_size 8 --lr 1e-3 --out runs/part1_baseline --checkpoint checkpoints/best_model.pth
```
 
### 4.3. Treinamento da Cabeça de Instâncias (Parte 2 — Trilha A Watershed)
Treinamento da U-Net 3-canais com Weighted Cross-Entropy para desacoplamento de fronteiras celulares:
```bash
python train.py --dataset dsb2018 --epochs 15 --batch_size 8 --lr 1e-3 --out_channels 3 --loss weighted_ce_3c --class_weights auto --out runs/part2_watershed --checkpoint checkpoints/part2_watershed.pth
```

### 4.4. Avaliação Comparativa em Lote
Avaliação com matching Hungarian IoU para cálculo de mAP@[.50:.95], contagem e geração de diagnósticos:
```bash
# Avaliar Baseline da Parte 1 (Componentes Conexos)
python evaluate.py --checkpoint checkpoints/best_model.pth --dataset dsb2018 --split val --matching hungarian --output-dir outputs/part1_eval

# Avaliar Trilha A da Parte 2 (Watershed)
python evaluate.py --checkpoint checkpoints/part2_watershed.pth --dataset dsb2018 --split val --matching hungarian --output-dir outputs/part2_eval
```

### 4.5. Execução das Ablações Sistemáticas (Parte 3 — Eixos 1 e 2)

O pipeline de ablações executa 2 seeds por configuração com pesos padronizados:

```bash
# Eixo 1: Recuperação de Resolução (U-Net Skips vs. Gargalo vs. DeepLabv3 ASPP)
PA1_CLASS_WEIGHTS="1.0,2.877,5.731" bash scripts/run_eixo1.sh
python scripts/summarize_eixo1.py

# Eixo 2: Funções de Perda e Fator γ (CE Ponderada vs. Focal γ=0, 1, 2, 5)
bash scripts/run_eixo2.sh
python scripts/summarize_eixo2.py
```

Tabelas consolidadas (média ± desvio e por seed), cálculo de campo receptivo teórico e curvas comparativas estão disponíveis diretamente em [`notebooks/inferencia.ipynb`](notebooks/inferencia.ipynb) e salvos em `outputs/part3_eixo1/` e `outputs/part3_eixo2/`.

### 4.6. Inferência em Mosaico com Fusão de Instâncias (Parte 4)

Compara o tiling ingênuo (center-crop e direct-stamp) com a costura via Hungarian matching local e Union-Find:

```bash
python scripts/run_mosaic_demo.py \
  --checkpoint checkpoints/part2_watershed.pth \
  --tile_size 256 \
  --stride 128 \
  --iou_overlap_threshold 0.20 \
  --output_dir outputs/part4_mosaic
```

Demonstração interativa, tabela quantitativa antes vs. depois e diagnósticos visuais de reconciliação de bordas estão disponíveis em [`notebooks/inferencia.ipynb`](notebooks/inferencia.ipynb).

---

## 5. Notebook de Inferência e Checkpoints

- **Inferência direta:** O notebook [`notebooks/inferencia.ipynb`](notebooks/inferencia.ipynb) atua como a vitrine técnica central do projeto. Ele implementa a função `predict_instances(image_path, model)` atendendo ao requisito oficial (*"recebe o caminho de uma imagem qualquer, devolve a máscara de instâncias colorida e a contagem. Roda sem retreinar"*). Por padrão, carrega o modelo oficial da Parte 2 (`checkpoints/part2_watershed.pth`), executando a decodificação Watershed instantaneamente na GPU/CPU.
- **Checkpoints disponíveis:**
  - `checkpoints/part0_synthetic.pth`: U-Net treinada nos dados sintéticos procedurais ($IoU > 0.99$).
  - `checkpoints/best_model.pth`: Baseline semântico binário da Parte 1 ($mAP = 0.4820$).
  - `checkpoints/part2_watershed.pth`: Modelo oficial da Parte 2 com cabeça Watershed ($mAP = 0.5157$).

## 6. Registro de Uso de Inteligência Artificial

Em conformidade com as diretrizes do assignment, os detalhes sobre como ferramentas de IA foram integradas durante o desenvolvimento do código, documentação e resolução de problemas estão documentados no arquivo [AI_LOG.md](AI_LOG.md).
