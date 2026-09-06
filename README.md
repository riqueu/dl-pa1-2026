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

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed placerat sem ex, sed eleifend tortor molestie sit amet. Vestibulum consequat maximus dignissim. Sed tincidunt eros sit amet arcu volutpat ornare.

---

## 4. Execução Rápida

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed placerat sem ex, sed eleifend tortor molestie sit amet. Vestibulum consequat maximus dignissim. Sed tincidunt eros sit amet arcu volutpat ornare.

---

## 5. Notebook de Inferência e Checkpoint

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed placerat sem ex, sed eleifend tortor molestie sit amet. Vestibulum consequat maximus dignissim. Sed tincidunt eros sit amet arcu volutpat ornare.

## 6. Registro de Uso de Inteligência Artificial

Em conformidade com as diretrizes do assignment, os detalhes sobre como ferramentas de IA foram integradas durante o desenvolvimento do código, documentação e resolução de problemas estão documentados no arquivo [AI_LOG.md](AI_LOG.md).