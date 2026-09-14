# Checkpoints do Modelo (Pesos .pth)

Conforme especificado no edital do PA1 (*"Pesos do modelo final (.pth) (link se for grande)"*), os arquivos de pesos possuem cerca de 99 MB cada e não são comitados diretamente no histórico do Git para evitar sobrecarregar o repositório.

## Checkpoints oficiais e dependência experimental

Temos exatamente **3 modelos pré-treinados**, um para cada fase inicial do trabalho:

1. **`part0_synthetic.pth` (Parte 0 — Teste de Sanidade Sintético):**
   - Configuração: U-Net com dados procedurais de elipses sobrepostas.
   - Desempenho: IoU Semântico = 0.9930 (treinado em < 5 min).
   - Link: [Download `part0_synthetic.pth`](https://github.com/riqueu/dl-pa1-2026/releases/download/v1.0.0/part0_synthetic.pth)

2. **`part1_baseline.pth` (Parte 1 — Baseline Semântico):**
   - Configuração: U-Net ResNet-34 binária (BCE + Soft Dice) com extração por componentes conexos.
   - Desempenho: mAP@[.50:.95] = 0.4820 | AP50 = 0.6611 | IoU Semântico = 0.8439.
   - Link: [Download `part1_baseline.pth`](https://github.com/riqueu/dl-pa1-2026/releases/download/v1.0.0/part1_baseline.pth)

3. **`part2_watershed.pth` (Parte 2 — Modelo Final Oficial de Instâncias):**
   - Configuração: U-Net ResNet-34 (3 canais: fundo, interior, fronteira; Weighted CE balanceada) com decodificação Watershed.
   - Desempenho: mAP@[.50:.95] = 0.5157 | AP50 = 0.7165 | Erro médio de contagem: 7.40 núcleos/img.
   - Link: [Download `part2_watershed.pth`](https://github.com/riqueu/dl-pa1-2026/releases/download/v1.0.0/part2_watershed.pth)
   - *Nota:* Este é o **modelo final oficial** utilizado como motor nas Partes 4 (Mosaico), 5 (Galeria de Falhas) e 6 (Teste de Estresse).

A Parte 6 também depende do checkpoint experimental
`checkpoints/part3_eixo1/deeplab_aspp_seed42.pth`, produzido pela configuração
`deeplab_aspp`, seed 42, do Eixo 1. Esse peso não está neste checkout nem entre
os três assets atuais da Release v1.0.0. Para reproduzir os resultados finais,
é necessário recuperar exatamente o arquivo da máquina que executou o teste ou
publicá-lo como asset adicional; pesos de outra execução não são equivalentes.

## Download via Linha de Comando

Para baixar o modelo oficial diretamente para esta pasta via terminal:
```bash
wget -P checkpoints/ https://github.com/riqueu/dl-pa1-2026/releases/download/v1.0.0/part2_watershed.pth
```
