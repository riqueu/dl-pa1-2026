# Resultados Oficiais da Parte 3 — Eixo 2 (Funções de Perda & Fator γ)

**Responsável:** Henrique Coelho Beltrão  
**Branch:** `feature/part3-eixo2-losses`  
**Dataset:** DSB2018 (Validação estratificada: 67 imagens)  
**Protocolo:** 15 épocas por modelo, 2 seeds aleatórias (`seed=42` e `seed=123`), avaliação via Hungarian Matching IoU.

---

## 1. Tabela Consolidada de Resultados (Média ± Desvio Padrão)

| Configuração | Perda | Fator $\gamma$ | mAP@[.50:.95] | AP @ IoU 0.50 | AP @ IoU 0.75 | Erro Médio Contagem | IoU Semântico | Dice Semântico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CE Ponderada** | `weighted_ce_3c` | — | **0.5002 ± 0.0155** | **0.7141 ± 0.0024** | **0.5579 ± 0.0149** | **7.25 ± 0.15** | **0.8134 ± 0.0102** | **0.8914 ± 0.0067** |
| **Focal ($\gamma=0$)** | `multiclass_focal` | 0.0 | **0.5015 ± 0.0205** | **0.7140 ± 0.0096** | **0.5635 ± 0.0198** | **7.07 ± 0.37** | **0.8083 ± 0.0139** | **0.8876 ± 0.0091** |
| **Focal ($\gamma=1$)** | `multiclass_focal` | 1.0 | 0.4830 ± 0.0506 | 0.6919 ± 0.0556 | 0.5363 ± 0.0609 | 7.31 ± 0.53 | 0.7860 ± 0.0345 | 0.8698 ± 0.0244 |
| **Focal ($\gamma=2$)** | `multiclass_focal` | 2.0 | 0.4723 ± 0.0092 | 0.7134 ± 0.0103 | 0.5208 ± 0.0166 | 7.47 ± 0.13 | 0.8002 ± 0.0020 | 0.8851 ± 0.0014 |
| **Focal ($\gamma=5$)** | `multiclass_focal` | 5.0 | 0.3636 ± 0.0059 | 0.6329 ± 0.0188 | 0.3800 ± 0.0032 | 10.32 ± 0.96 | 0.7533 ± 0.0060 | 0.8536 ± 0.0035 |

---

## 2. Detalhamento por Seed

### 2.1. Seed 42
* `ce_weighted`: mAP: **0.5157** | AP50: 0.7165 | AP75: 0.5727 | Erro Cont.: 7.40 | IoU: 0.8236
* `focal_gamma0`: mAP: **0.5220** | AP50: 0.7236 | AP75: 0.5832 | Erro Cont.: 7.43 | IoU: 0.8222
* `focal_gamma1`: mAP: 0.4324 | AP50: 0.6364 | AP75: 0.4754 | Erro Cont.: 7.84 | IoU: 0.7515
* `focal_gamma2`: mAP: 0.4632 | AP50: 0.7032 | AP75: 0.5042 | Erro Cont.: 7.34 | IoU: 0.7982
* `focal_gamma5`: mAP: 0.3577 | AP50: 0.6141 | AP75: 0.3833 | Erro Cont.: 11.28 | IoU: 0.7593

### 2.2. Seed 123
* `ce_weighted`: mAP: **0.4847** | AP50: 0.7117 | AP75: 0.5430 | Erro Cont.: 7.10 | IoU: 0.8033
* `focal_gamma0`: mAP: **0.4810** | AP50: 0.7045 | AP75: 0.5437 | Erro Cont.: 6.70 | IoU: 0.7944
* `focal_gamma1`: mAP: **0.5336** | AP50: 0.7475 | AP75: 0.5972 | Erro Cont.: 6.78 | IoU: 0.8205
* `focal_gamma2`: mAP: 0.4815 | AP50: 0.7237 | AP75: 0.5374 | Erro Cont.: 7.60 | IoU: 0.8022
* `focal_gamma5`: mAP: 0.3696 | AP50: 0.6517 | AP75: 0.3768 | Erro Cont.: 9.36 | IoU: 0.7473

---

## 3. Discussão Conceitual e Resposta ao Edital

O edital questiona o efeito do desbalanceamento severo da classe fronteira ($2.8\%$ dos pixels) quando variamos o parâmetro $\gamma \in \{0, 1, 2, 5\}$. Nossos experimentos revelam três descobertas fundamentais:

1. **Equivalência e Pico de Desempenho em $\gamma = 0$:**
   * A Cross-Entropy Ponderada (`weighted_ce_3c`) e a Focal Loss com $\gamma = 0.0$ apresentaram desempenhos praticamente idênticos ($mAP \approx 0.501$), confirmando numericamente a equivalência matemática entre as formulações.
   * Ambas atingiram a menor taxa de erro de contagem ($~7.0$ a $7.2$ núcleos/imagem) e o maior AP@0.50 ($> 0.714$).

2. **Degradação Monótona para $\gamma$ Elevado:**
   * À medida que $\gamma$ cresce para $2.0$ e $5.0$, o desempenho cai progressivamente ($mAP$ desaba de $0.501 \to 0.364$, e o erro de contagem aumenta em $+46\%$ para $10.32$).

3. **Por que a Focal Loss Agressiva Prejudica a Decodificação por Watershed?**
   * No watershed, a identificação dos **marcadores de interior** é tão ou mais importante que a crista de fronteira.
   * Com $\gamma=5$, o termo modulador $(1 - p_t)^5$ atenua exponencialmente o gradiente de pixels "fáceis" (como o interior do núcleo, onde a rede rapidamente atinge confiança $p_t \approx 0.85$, resultando em fator de escala $(0.15)^5 \approx 0.00007$).
   * Isso enfraquece o sinal de aprendizado nos interiores celulares, gerando sementes fragmentadas ou ausentes. Por outro lado, a rede hiper-foca no ruído das bordas ambíguas, desestabilizando as bacias de inundação do Watershed.

**Conclusão Técnica para a Apresentação:**  
Para a Trilha A (Fronteiras e Watershed), **o balanceamento estático de classes via frequência inversa suavizada ($\alpha$) é a abordagem ótima**, sendo superior à modulação dinâmica por $\gamma$ elevado.
