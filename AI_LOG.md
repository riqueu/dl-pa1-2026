# Registro de Uso de IA (AI_LOG)

Uso de ferramentas de IA como suporte de pair programming, planejamento e divisão de atividades da dupla (`docs/`), automação de scripts e revisão teórica. Todas as decisões técnicas, implementações e interpretações dos resultados foram validadas e dominadas pela dupla.

---

### Episódio 1: Planejamento, Divisão de Tarefas e Pipeline Base (Partes 0, 1 e 2)
- **Problema:** Estruturar o repositório, planejar a divisão de tarefas colaborativas entre a dupla sem conflitos de merge e implementar a U-Net base com métricas de matching 1-para-1 sem bibliotecas prontas.
- **Aporte da IA:** Elaboração dos roteiros de divisão de atividades em `docs/` (definindo contratos de tensores, branches paralelas e responsabilidades dos membros A e B), além da vetorização do matching bipartite húngaro e cálculo de mAP@[.50:.95].
- **Decisão da dupla:** Adoção de desenvolvimento desacoplado por interfaces, implementação 100% autoral das perdas e do decoder, e fixação dos splits estratificados em disco (`data/splits.json`) para reprodutibilidade.

---

### Episódio 2: Ablações de Resolução e Funções de Perda (Parte 3)
- **Problema:** Comparar mecanismos de resolução (U-Net Skips vs. Gargalo Cego vs. cabeça DeepLab/ASPP autoral, inspirada no DeepLabv3) e sensibilidade a perdas (CE balanceada vs. Focal $\gamma \in \{0,1,2,5\}$) com 2 seeds cada.
- **Aporte da IA:** Planejamento experimental em `docs/parte3.md`, automação dos runners de grade (`run_eixo1.sh`, `run_eixo2.py`) e ajuste das dilatações atrous com output stride 16 na ResNet34.
- **Decisão da dupla:** Manter a U-Net com CE balanceada ($\gamma=0$) como arquitetura oficial. A tendência observada é compatível com maior fragilidade do ASPP em núcleos pequenos; o experimento não isola causalidade. A configuração $\gamma=5$ também apresentou piora nas cristas de fronteira.

---

### Episódio 3: Inferência em Mosaico e Costura de Instâncias (Parte 4)
- **Problema:** Processar imagens grandes via janelas deslizantes ($256 \times 256$, stride 128) e corrigir a fragmentação/duplicação de células na linha de corte entre tiles.
- **Aporte da IA:** Divisão de frentes em `docs/parte4.md` (Membro A: tiling/diagnóstico; Membro B: stitching), lógica de fatiamento com padding e desenho conceitual da reconciliação de instâncias.
- **Decisão da dupla:** Desenvolver algoritmo autoral de costura via Hungarian matching local e fusão transitiva por Union-Find, elevando o mAP em $+5{,}15\text{ p.p.}$ e corrigindo a contagem.

---

### Episódio 4: Galeria de Falhas e Correção Morfológica (Parte 5)
- **Problema:** Mapear os 5 piores casos de falha do modelo oficial, confrontar o campo receptivo teórico ($RF$) com a morfologia celular e propor uma intervenção eficaz.
- **Aporte da IA:** Estruturação do plano em `docs/parte5.md`, script de mineração de falhas e comparação do campo receptivo teórico com os diâmetros medidos após redimensionamento para $256\times256$.
- **Decisão da dupla:** Implementar uma correção morfológica com parâmetros fixos calibrados para o Caso 2 (`seed_closing_radius=4`). O erro de contagem foi zerado nesse caso, mas o mAP final de $0{,}2518$ mostra que as máscaras permanecem imperfeitas.

---

### Episódio 5: Teste de Estresse de Escala e Análise Teórica (Parte 6)
- **Problema:** Avaliar o comportamento de U-Net e da cabeça DeepLab/ASPP autoral, inspirada no DeepLabv3, sob sub-resolução ($0{,}5\times$) e sobre-resolução ($2{,}0\times$).
- **Aporte da IA:** Estruturação do plano em `docs/parte6.md`, formalização teórica da não-invariância de escala em FCNs e automação do pipeline multiescala.
- **Decisão da dupla:** Congelar limiares de inferência para avaliar a robustez crua da representação. A queda de $88{,}9\%$ do ASPP em $0{,}5\times$ é compatível com maior sensibilidade a núcleos subamostrados, enquanto a U-Net reteve mais desempenho; trata-se de interpretação dos resultados, não de prova causal isolada.
