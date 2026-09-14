# Plano de Implementação da Parte 5 — Galeria de Falhas e Diagnósticos

Este documento estabelece o plano técnico, os contratos de código e as especificações para a execução da **Parte 5 — Galeria de Falhas** do PA1 (Deep Learning — FGV EMAp).

**Responsável:** Membro A (Henrique Coelho Beltrão)  
**Branch de Desenvolvimento:** `feature/part5-failure-gallery`  
**Dataset:** Data Science Bowl 2018 (DSB2018) — Split de validação (67 imagens)  
**Modelo Avaliado:** Checkpoint oficial da Parte 2 (`checkpoints/part2_watershed.pth`)

---

## 1. Requisitos do Edital (`PA1.pdf`)

O edital estipula os seguintes requisitos mandatórios para a Parte 5:
1. **Cinco Imagens Críticas de Erro Expressivo:**
   - Selecionar 5 imagens onde o modelo final falha categoricamente (baixo mAP e/ou erro expressivo de contagem).
   - Para cada caso, gerar a figura com 4 painéis:
     1. Imagem de entrada;
     2. Ground Truth de instâncias;
     3. Predição do modelo;
     4. Mapa intermediário relevante (probabilidade de fronteira e interior celular).
2. **Diagnóstico Formal de Falha:**
   - Elaborar hipótese biológica, óptica ou arquitetural fundamentada para cada um dos 5 casos (ex.: contraste ambíguo, diâmetro que excede o raio de inundação do Watershed, aglomerados densos com erosão insuficiente).
3. **Cálculo de Campo Receptivo vs. Distribuição de Tamanhos:**
   - Calcular a distribuição empírica de diâmetros e áreas de todos os núcleos do dataset DSB2018 (histograma de diâmetros equivalentes $d = 2\sqrt{A/\pi}$).
   - Redimensionar cada máscara para $256\times256$ antes de medir os objetos, mantendo diâmetros e campo receptivo na mesma escala.
   - Comparar o encoder OS16 sem atrous ($739\text{ px}$) e com atrous ($931\text{ px}$), mantendo a mesma resolução de saída; os ramos ASPP alcançam teoricamente $1123$ a $1507\text{ px}$.
   - Explicar o que a relação entre campo receptivo e tamanho dos núcleos revela sobre as causas reais de erro do modelo.
4. **Implementação de uma Correção (Antes vs. Depois):**
   - Escolher uma falha diagnosticável e implementar uma correção no pipeline. A execução final usou parâmetros fixos calibrados especificamente para o Caso 2.
   - Demonstrar quantitativamente e visualmente a recuperação do caso (Antes vs. Depois).

---

## 2. Passo a Passo Técnico da Implementação

### 2.1. Script de Rastreio Automatizado dos 5 Piores Casos (`scripts/run_failure_gallery.py`)
- Executa a inferência de `checkpoints/part2_watershed.pth` nas 67 imagens do split de validação.
- Calcula o mAP@[.50:.95] e o erro absoluto de contagem $|N_{\text{pred}} - N_{\text{gt}}|$ para cada imagem.
- Ordena as imagens por severidade de erro e seleciona os 5 casos mais representativos cobrindo diferentes perfis de falha:
  1. *Aglomerado Ultra-Denso com Fusão:* Núcleos muito pequenos encostados onde o interior colapsou.
  2. *Células Gigantes / Alargadas:* Núcleos muito extensos cujo interior sofre sub-segmentação ou fragmentação interna.
  3. *Baixo Contraste / Borda Esmaecida:* Fundo irregular com crista de fronteira incompleta que vaza a bacia de inundação.
  4. *Falso Positivo de Textura:* Detritos de fundo interpretados espuriamente como núcleos celulares.
  5. *Anotação Ruidosa ou Ambígua no Ground Truth:* Limites biológicos inconsistentes na marcação original.

### 2.2. Geração dos Painéis Diagnósticos
Para cada imagem selecionada, salvar em `outputs/part5_gallery/` um painel em alta resolução (`failure_case_1.png` a `failure_case_5.png`):
- Painel 1: Imagem original RGB com zoom na região do erro.
- Painel 2: Máscara Ground Truth colorizada.
- Painel 3: Predição Final do Watershed colorizada (com contagem predita vs. real).
- Painel 4: Mapa Intermediário de Probabilidades (RGB: Vermelho = Fronteira, Verde = Interior, Azul = Fundo).

### 2.3. Levantamento Empírico do Tamanho dos Núcleos
- Iterar sobre todas as máscaras individuais do DSB2018 (`stage1_train`), redimensioná-las por vizinho mais próximo para $256\times256$ e calcular:
  $$\text{Área } A = \sum \text{pixels}, \quad \text{Diâmetro Equivalente } d = 2 \sqrt{\frac{A}{\pi}}$$
- Gerar gráfico `outputs/part5_gallery/nuclei_size_distribution.png`:
  - Histograma com curva de densidade dos diâmetros celulares.
  - Linhas indicando média ($13{,}91$ px), percentil 95 ($32{,}31$ px) e máximo ($83{,}57$ px) na escala efetivamente recebida pela rede.
  - Das 29.461 máscaras originais, 18 objetos minúsculos colapsam para área zero após o redimensionamento, evidenciando perda real de informação na entrada.
  - Comparação OS16 sem e com atrous, evitando confundir mudança de dilatação com mudança de resolução de saída.
- **Discussão Conceitual:** O campo receptivo teórico pode exceder $256$ px porque inclui posições preenchidas por padding. O contexto observável permanece limitado à entrada $256\times256$, e o campo receptivo efetivo aprendido tende a ser menor. Portanto, a relação com o diâmetro dos núcleos sustenta, mas não comprova isoladamente, a hipótese de que parte das falhas decorra de perda de detalhes finos de fronteira.

### 2.4. Implementação e Medição da Correção
- **Correção executada:** *Correção morfológica calibrada para o Caso 2*:
  - Usar os parâmetros fixos `interior_threshold=0.40`, `foreground_threshold=0.40`, `min_area=60` e `seed_closing_radius=4`.
  - Comparar o mAP e o erro de contagem no caso selecionado antes e depois da correção.
  - Explicitar que erro de contagem zero não implica segmentação perfeita: o mAP após a correção permanece em $0{,}2518$.
  - Salvar painel visual `outputs/part5_gallery/correction_before_after.png`.

---

## 3. Entregáveis Esperados

Ao término da implementação do Membro A, a branch `feature/part5-failure-gallery` deverá conter:
1. `scripts/run_failure_gallery.py`: Script autônomo e reproduzível.
2. `outputs/part5_gallery/`:
   - `failure_case_1.png` a `failure_case_5.png` (painéis das 5 falhas).
   - `nuclei_size_distribution.png` (distribuição de tamanhos vs. campo receptivo).
   - `correction_before_after.png` (evidência visual da correção).
   - `gallery_metrics.json` (tabela de métricas dos casos críticos antes e depois).
3. Testes ou validação em `tests/test_gallery.py` (se aplicável).
