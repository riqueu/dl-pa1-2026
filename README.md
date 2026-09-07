# PA1 - Segmentação de Instâncias com Arquiteturas Densas

Repositório para o Programming Assignment 1 (PA1) da disciplina de Aprendizado Profundo (FGV EMAp).

## Dupla
- [Henrique Coelho Beltrão](https://github.com/riqueu)
- [Isaias Gouvêa Gonçalves](https://github.com/isaiasgoncalves)

---

## 1. Visão Geral do Projeto

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed placerat sem ex, sed eleifend tortor molestie sit amet. Vestibulum consequat maximus dignissim. Sed tincidunt eros sit amet arcu volutpat ornare.

```bash
dl-pa1-2026
├── AI_LOG.md
├── checkpoints/
├── data/
├── evaluate.py            # Avaliação em lote no conjunto de teste/validação
├── LICENSE
├── notebooks/
│   ├── exploratory.ipynb  # Prototipação e inspeção de dados
│   └── inferencia.ipynb   # Notebook obrigatório de inferência
├── PA1.pdf
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py         # Datasets reais, gerador sintético (elipses) e splits
│   ├── __init__.py
│   ├── losses.py          # CE balanceada, Focal Loss, L1/L2
│   ├── metrics.py         # Matching Hungarian/Greedy, cálculo de mAP@[.50:.95] e erro de contagem
│   ├── models.py          # Encoders/Decoders (U-Net, ResUNet, SegNet, DeepLab/ASPP)
│   ├── postprocess.py     # Componentes conexos, Watershed / Centros+Offsets / Embeddings
│   └── utils.py           # Helpers de visualização, cálculo do campo receptivo e tiling
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

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed placerat sem ex, sed eleifend tortor molestie sit amet. Vestibulum consequat maximus dignissim. Sed tincidunt eros sit amet arcu volutpat ornare.

---

## 5. Notebook de Inferência e Checkpoint

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed placerat sem ex, sed eleifend tortor molestie sit amet. Vestibulum consequat maximus dignissim. Sed tincidunt eros sit amet arcu volutpat ornare.

## 6. Registro de Uso de Inteligência Artificial

Em conformidade com as diretrizes do assignment, os detalhes sobre como ferramentas de IA foram integradas durante o desenvolvimento do código, documentação e resolução de problemas estão documentados no arquivo [AI_LOG.md](AI_LOG.md).