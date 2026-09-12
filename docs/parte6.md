# Plano de Implementação da Parte 6 — Teste de Estresse (Mudança de Escala)

Este documento estabelece o plano técnico, os contratos de código, o protocolo experimental e a fundamentação teórica formal para a execução da **Parte 6 — Teste de Estresse (Opção 3: Mudança de Escala)** do PA1 (Deep Learning — FGV EMAp).

**Responsável:** Membro B (Isaías Silveira da Silva)  
**Branch de Desenvolvimento:** `feature/part6-scale-stress`  
**Dataset:** Data Science Bowl 2018 (DSB2018) — Split de validação (67 imagens)  
**Modelos Avaliados:** 
1. U-Net ResNet-34 + Watershed (`checkpoints/part2_watershed.pth`)
2. DeepLabv3+ ResNet-34 + ASPP (`checkpoints/part3_eixo1/deeplab_aspp_seed42.pth`)

---

## 1. Requisitos do Edital (`PA1.pdf`)

O edital estipula os seguintes requisitos para a Parte 6 (Opção 3):
1. **Avaliação sob Mudança de Escala:**
   - Avaliar os modelos em $0{,}5\times$, $1{,}0\times$ (escala nativa) e $2{,}0\times$.
   - Quantificar o impacto nas métricas oficiais de instância: mAP@[.50:.95], AP50, AP75 e Erro Absoluto de Contagem.
   - Gerar a curva de degradação do mAP em função do fator de escala.
2. **Comparação de Arquiteturas:**
   - Contrastar a resiliência da arquitetura U-Net (com skip connections multinível) contra a arquitetura DeepLabv3 (com módulo ASPP — *Atrous Spatial Pyramid Pooling*).
3. **Fundamentação Teórica Formal:**
   - Responder rigorosamente: **Por que uma rede totalmente convolucional (FCN) não é invariante a escala?**
   - Responder rigorosamente: **O que o ASPP faz (ou não faz) a respeito?**

---

## 2. Passo a Passo Técnico da Implementação

### 2.1. Script de Avaliação Automatizada (`scripts/run_scale_stress.py`)
O script deve implementar o seguinte pipeline:
1. Carregar os dois checkpoints treinados (`part2_watershed.pth` e `deeplab_aspp_seed42.pth`).
2. Para cada fator de escala $s \in \{0{,}5,\, 1{,}0,\, 2{,}0\}$:
   - Redimensionar as imagens de validação para $(H' = \text{round}(H \cdot s),\, W' = \text{round}(W \cdot s))$ via interpolação bilinear.
   - Executar o *forward pass* para obter os mapas de logits.
   - Aplicar a decodificação Watershed na resolução escalada para extrair os marcadores e instâncias.
   - Redimensionar a máscara de instâncias resultante de volta para a dimensão nativa $(H, W)$ via interpolação por vizinho mais próximo (`nearest`), preservando os IDs discretos de instância.
   - Computar as métricas de matching Hungarian / IoU contra o Ground Truth nativo $(H, W)$.
3. Salvar as métricas consolidadas em `outputs/part6_stress/scale_metrics.json`.

### 2.2. Geração da Curva de Degradação (`outputs/part6_stress/scale_degradation_curve.png`)
- Gráfico com dois eixos:
  - Eixo X: Fator de Escala ($0{,}5\times$, $1{,}0\times$, $2{,}0\times$ em escala logarítmica ou categórica).
  - Eixo Y: mAP@[.50:.95] e AP50.
- Duas curvas sobrepostas:
  - Curva azul: U-Net ResNet-34 (Baseline Parte 2).
  - Curva laranja: DeepLabv3 ASPP (Parte 3 Eixo 1).
- Visualização clara da taxa percentual de queda de desempenho em relação à escala nativa $1{,}0\times$.

### 2.3. Painel Visual Comparativo (`outputs/part6_stress/scale_visual_comparison.png`)
- Grade visual ilustrando um caso desafiador do conjunto de validação sob as três escalas:
  - Linha 1: Imagem de entrada redimensionada ($0{,}5\times$, $1{,}0\times$, $2{,}0\times$).
  - Linha 2: Predição de instâncias pela U-Net Watershed.
  - Linha 3: Predição de instâncias pelo DeepLabv3 ASPP.
  - Linha 4: Ground Truth de referência.

---

## 3. Fundamentação Teórica Obrigatória

### 3.1. Por que uma FCN não é invariante a escala?
1. **Equivariância a Translação vs. Não-Equivariância a Escala:**
   - A operação de convolução 2D contínua é equivariante a translações do grupo aditivo $(\mathbb{R}^2, +)$:
     $$\mathcal{T}_{\mathbf{v}}[f](\mathbf{x}) = f(\mathbf{x} - \mathbf{v}) \implies (f * k)(\mathbf{x} - \mathbf{v}) = (\mathcal{T}_{\mathbf{v}}[f] * k)(\mathbf{x})$$
   - No entanto, considere o operador de escala $\mathcal{S}_{\sigma}[f](\mathbf{x}) = f(\mathbf{x} / \sigma)$ para $\sigma > 0$. A convolução com um kernel fixo $k(\mathbf{x})$ satisfaz:
     $$(\mathcal{S}_{\sigma}[f] * k)(\mathbf{x}) = \int_{\mathbb{R}^2} f\left(\frac{\mathbf{y}}{\sigma}\right) k(\mathbf{x} - \mathbf{y})\, d\mathbf{y} = \sigma^2 \int_{\mathbb{R}^2} f(\mathbf{z})\, k(\mathbf{x} - \sigma \mathbf{z})\, d\mathbf{z}$$
   - Para ser equivariante a escala, seria necessário que o kernel também fosse escalado por $k_\sigma(\mathbf{u}) = \frac{1}{\sigma^2} k(\mathbf{u}/\sigma)$. Como os pesos aprendidos em tensores discretos possuem **suporte espacial fixo** (ex.: $3 \times 3$ pixels com passo de amostragem de grade unitária), os filtros da FCN não se adaptam à variação de frequência espacial.
2. **Impacto Físico no Problema Biológico:**
   - **Em $0{,}5\times$ (Redução):** O diâmetro médio dos núcleos cai de $\approx 25\text{ px}$ para $\approx 12{,}5\text{ px}$. A área celular encolhe $4\times$. As fronteiras celulares de 1 pixel entre núcleos contíguos são sub-amostradas e colapsam para menos de 1 pixel (efeito de aliasing/apagamento de crista), impedindo a separação por Watershed e provocando fusão massiva (under-segmentation).
   - **Em $2{,}0\times$ (Ampliação):** O diâmetro médio sobe para $\approx 50\text{ px}$. Os filtros $3 \times 3$ passam a enxergar apenas variações de textura intracelular e gradientes cromáticos esmaecidos, em vez da curvatura global da membrana celular, provocando quebra de núcleos em múltiplos fragmentos (over-segmentation).

### 3.2. O que o ASPP faz (ou não faz) a respeito?
1. **O que o ASPP FAZ:**
   - O *Atrous Spatial Pyramid Pooling* introduz convoluções dilatadas em paralelo com diferentes taxas de amostragem $r \in \{1, 6, 12, 18\}$ sobre o mesmo mapa de características de *output stride* 16 ($OS=16$).
   - O campo receptivo efetivo de cada ramo é $RF_r = K + (K - 1)(r - 1)$. Para $K=3$:
     - $r=6 \implies RF = 13$ neurônios do mapa base;
     - $r=12 \implies RF = 25$ neurônios do mapa base;
     - $r=18 \implies RF = 37$ neurônios do mapa base.
   - Isso permite ao modelo sondar contextos de múltiplos diâmetros simultaneamente, amostrando o mesmo tensor em diferentes frequências espaciais sem decimar a resolução com *stride*.
2. **O que o ASPP NÃO FAZ:**
   - **Não confere invariância contínua a escala:** O ASPP atua apenas em um **gargalo intermediário discreto** (após o Stage 4 do ResNet) e com taxas de dilatação fixas $\{6, 12, 18\}$. Ele não adapta seus pesos dinamicamente para escalas contínuas arbitrárias.
   - **Não resolve a perda de frequências finas na entrada:** Se a imagem de entrada é reduzida ($0{,}5\times$) e as fronteiras de 1 pixel são eliminadas na amostragem inicial, nenhum ramo do ASPP consegue recuperar essa informação perdida, pois todos operam sobre representações já decimadas.
   - **Susceptibilidade a Gridding:** Em dilatações altas ($r=18$), os pesos amostram pixels esparsos sem cobrir os vizinhos imediatos, o que em imagens ampliadas ($2{,}0\times$) pode gerar ruído amostral na transição núcleo-fundo.

---

## 4. Entregáveis Esperados

Ao término da implementação do Membro B, a branch `feature/part6-scale-stress` deverá conter:
1. `scripts/run_scale_stress.py`: Script autônomo de estresse de escala.
2. `outputs/part6_stress/`:
   - `scale_metrics.json` (tabela quantitativa completa com mAP, AP50, AP75 e erro de contagem para $0{,}5\times$, $1{,}0\times$ e $2{,}0\times$).
   - `scale_degradation_curve.png` (curva de degradação comparativa U-Net vs. DeepLabv3 ASPP).
   - `scale_visual_comparison.png` (painel visual comparativo).
3. Resposta teórica incorporada à apresentação e documentada no `AI_LOG.md` e `inferencia.ipynb`.

