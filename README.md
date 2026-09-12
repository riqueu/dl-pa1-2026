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

| Abordagem | mAP@[.50:.95] | AP @ IoU 0.50 | Erro Médio Contagem | IoU Semântico |
| :--- | :---: | :---: | :---: | :---: |
| **Parte 1: Baseline Semântico** (Componentes Conexos) | 0.4820 | 0.6611 | 8.55 núcleos/img | **0.8439** |
| **Parte 2: Trilha A** (Fronteiras e Watershed) | **0.5157** | **0.7165** | **7.40 núcleos/img** | 0.8236 |
| *Caso Crítico Denso (369 núcleos) — Baseline* | *0.1443* | *—* | *174 núcleos (fusão)* | *—* |
| *Caso Crítico Denso (369 núcleos) — Trilha A* | **0.3374** | *—* | **47 núcleos (-73%)** | *—* |

```bash
dl-pa1-2026
├── AI_LOG.md              # Registro conciso de assistência de IA
├── checkpoints/           # Pesos dos modelos treinados (.pth)
├── data/                  # DSB2018 stage1_train e splits estratificados
├── docs/                  # Planos técnicos e divisão de tarefas para a dupla
│   ├── parte0e1.md        # Especificação das Partes 0 e 1
│   ├── parte2.md          # Especificação da Trilha A (Watershed)
│   ├── parte3.md          # Especificação das Ablações (Eixos 1 e 2)
│   └── parte4.md          # Especificação de Mosaico e Stitching
├── evaluate.py            # Avaliação em lote no conjunto de teste/validação
├── LICENSE
├── notebooks/
│   ├── exploratory.ipynb  # Prototipação e inspeção de dados
│   └── inferencia.ipynb   # Notebook obrigatório e vitrine central do projeto
├── PA1.pdf
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py         # Datasets reais, gerador sintético e alvos 3-classes
│   ├── __init__.py
│   ├── losses.py          # CE balanceada, Focal Loss multiclasse e pesos
│   ├── metrics.py         # Matching Hungarian/Greedy e cálculo de mAP@[.50:.95]
│   ├── models.py          # U-Net autoral com encoder ResNet34
│   ├── postprocess.py     # Componentes conexos e decodificação Watershed
│   └── utils.py           # Colorização, overlay e gráficos diagnósticos
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

### 4.5. Inferência em Mosaico com Fusão de Instâncias

Compara o tiling ingênuo com a costura por IoU na faixa de sobreposição:

```bash
python scripts/run_mosaic_demo.py --checkpoint checkpoints/part2_watershed.pth --tile_size 256 --stride 128 --iou_overlap_threshold 0.20 --output_dir outputs/part4_mosaic
```

---

## 5. Notebook de Inferência e Checkpoints

- **Inferência direta:** O notebook [`notebooks/inferencia.ipynb`](notebooks/inferencia.ipynb) atua como a vitrine técnica central do projeto. Ele implementa a função `predict_instances(image_path, model)` atendendo ao requisito oficial (*"recebe o caminho de uma imagem qualquer, devolve a máscara de instâncias colorida e a contagem. Roda sem retreinar"*). Por padrão, carrega o modelo oficial da Parte 2 (`checkpoints/part2_watershed.pth`), executando a decodificação Watershed instantaneamente na GPU/CPU.
- **Checkpoints disponíveis:**
  - `checkpoints/part0_synthetic.pth`: U-Net treinada nos dados sintéticos procedurais ($IoU > 0.99$).
  - `checkpoints/best_model.pth`: Baseline semântico binário da Parte 1 ($mAP = 0.4820$).
  - `checkpoints/part2_watershed.pth`: Modelo oficial da Parte 2 com cabeça Watershed ($mAP = 0.5157$).

## 6. Registro de Uso de Inteligência Artificial

Em conformidade com as diretrizes do assignment, os detalhes sobre como ferramentas de IA foram integradas durante o desenvolvimento do código, documentação e resolução de problemas estão documentados no arquivo [AI_LOG.md](AI_LOG.md).
