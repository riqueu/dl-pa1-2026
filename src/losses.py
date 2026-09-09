"""CE balanceada, Focal Loss, L1/L2.

Perdas da baseline binária (Partes 0 e 1) e da ablação do Eixo 2 (Parte 3).

Todas as perdas compartilham a mesma assinatura `(logits, target)`, com
logits float32 (B, 1, H, W) crus e target float32 (B, 1, H, W) em {0.0, 1.0}.
Assinatura única é o que permite trocar de perda por flag de CLI, sem editar
o loop de treino.

Glossário:
- perda: número que mede o quanto a rede errou; o treino tenta diminuí-lo.
- logits: nota crua da rede, de -inf a +inf; só vira probabilidade depois do sigmoid.
- BCE: olha cada pixel isolado e castiga conforme a distância entre nota e resposta certa.
- Dice: mede o quanto a área prevista e a área real se sobrepõem; ignora acerto no fundo.
- Soft Dice: mesma conta, usando a probabilidade em vez de sim/não, para poder treinar.
- desbalanceamento: quando uma classe ocupa quase toda a imagem e a outra quase nada.
- pos_weight: multiplicador que aumenta o castigo por errar a classe rara.
- alpha: na focal, quanto peso dar à classe rara.
- gamma: na focal, o quanto ignorar os exemplos que a rede já acerta com folga.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Dice diferenciável sobre as probabilidades, calculado por imagem.

    Otimiza sobreposição de região em vez de acerto pixel a pixel, o que a torna
    naturalmente robusta ao desbalanceamento entre fundo e objeto.

    O coeficiente é calculado imagem a imagem e só depois promediado no lote.
    Calcular um Dice único sobre o lote inteiro deixaria as imagens fáceis
    mascararem as difíceis.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        """Inicializa a perda.

        Args:
            smooth: Constante somada ao numerador e ao denominador. Além de evitar
                divisão por zero, faz uma imagem sem objeto e sem predição pontuar 1.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcula 1 - Dice médio.

        Args:
            logits: Tensor (B, 1, H, W) com logits crus.
            target: Tensor (B, 1, H, W) binário em {0.0, 1.0}.

        Returns:
            Escalar com a perda.
        """
        probs = torch.sigmoid(logits)

        # Achata cada imagem, preservando o eixo do lote.
        probs_flat = probs.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combinação de BCE com logits e Soft Dice: a perda da baseline.

    A BCE dá gradiente estável e bem condicionado por pixel; o Dice corrige a
    tendência da BCE de favorecer a classe majoritária. Somar as duas é prática
    padrão em segmentação binária desbalanceada.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: Optional[float] = None,
        smooth: float = 1.0,
    ) -> None:
        """Inicializa a perda combinada.

        Args:
            bce_weight: Peso do termo de BCE.
            dice_weight: Peso do termo de Dice.
            pos_weight: Peso da classe positiva na BCE, tipicamente a razão
                fundo/frente do conjunto de treino. None desliga a reponderação.
            smooth: Constante de suavização repassada ao Dice.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcula a soma ponderada de BCE e Dice.

        Args:
            logits: Tensor (B, 1, H, W) com logits crus.
            target: Tensor (B, 1, H, W) binário em {0.0, 1.0}.

        Returns:
            Escalar com a perda.
        """
        pw = None
        if self.pos_weight is not None:
            # Criado no forward para acompanhar device e dtype dos logits.
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, target)


class FocalLoss(nn.Module):
    """Focal Loss binária, parametrizada por alpha e gamma (Eixo 2 da Parte 3).

    Multiplica a entropia cruzada por `(1 - p_t) ** gamma`, reduzindo o peso dos
    exemplos que a rede já acerta com folga e concentrando o gradiente nos
    difíceis. Com `gamma=0` e `alpha=None` reproduz exatamente a BCE.

    A implementação usa `logsigmoid` em vez de `log(sigmoid(x))` para evitar
    underflow: com logits muito grandes o sigmoid arredonda para 0 ou 1, e o log
    de 0 vira infinito. A versão fundida faz a conta sem passar por esse ponto.
    """

    def __init__(self, alpha: Optional[float] = 0.25, gamma: float = 2.0) -> None:
        """Inicializa a perda.

        Args:
            alpha: Peso da classe positiva em [0, 1]; a negativa recebe 1 - alpha.
                None desliga o balanceamento por classe.
            gamma: Expoente do fator de foco. 0 desliga o foco.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcula a focal loss média.

        Args:
            logits: Tensor (B, 1, H, W) com logits crus.
            target: Tensor (B, 1, H, W) binário em {0.0, 1.0}.

        Returns:
            Escalar com a perda.
        """
        # -log(p) e -log(1 - p), ambos numericamente estáveis.
        nll_pos = -F.logsigmoid(logits)
        nll_neg = -F.logsigmoid(-logits)

        probs = torch.sigmoid(logits)

        # Fator de foco: (1 - p) para o alvo positivo, p para o negativo.
        focus_pos = (1.0 - probs).pow(self.gamma)
        focus_neg = probs.pow(self.gamma)

        w_pos, w_neg = 1.0, 1.0
        if self.alpha is not None:
            w_pos, w_neg = self.alpha, 1.0 - self.alpha

        loss = (
            w_pos * target * focus_pos * nll_pos
            + w_neg * (1.0 - target) * focus_neg * nll_neg
        )
        return loss.mean()


def compute_pos_weight(masks: torch.Tensor, eps: float = 1e-7) -> float:
    """Calcula a razão fundo/frente para uso como `pos_weight` na BCE.

    Deve ser calculada uma única vez, sobre o split de treino, e nunca sobre
    validação ou teste.

    Args:
        masks: Tensor com máscaras semânticas binárias, de qualquer shape.
        eps: Estabilizador para o caso de não haver nenhum pixel positivo.

    Returns:
        Razão entre pixels de fundo e pixels de objeto.
    """
    positives = masks.sum().item()
    total = masks.numel()
    return float((total - positives) / (positives + eps))


def build_loss(name: str = "bce_dice", **kwargs: Any) -> nn.Module:
    """Constrói a perda pelo nome, para seleção via linha de comando.

    Args:
        name: 'bce_dice', 'bce', 'dice' ou 'focal'.
        **kwargs: Repassados ao construtor da perda escolhida.

    Returns:
        Módulo de perda com assinatura `(logits, target)`.

    Raises:
        ValueError: Se o nome não for reconhecido.
    """
    registry: Dict[str, Any] = {
        "bce_dice": BCEDiceLoss,
        "bce": lambda **kw: BCEDiceLoss(bce_weight=1.0, dice_weight=0.0, **kw),
        "dice": lambda **kw: BCEDiceLoss(bce_weight=0.0, dice_weight=1.0, **kw),
        "focal": FocalLoss,
    }

    if name not in registry:
        raise ValueError(f"Perda '{name}' desconhecida. Disponíveis: {sorted(registry)}.")

    return registry[name](**kwargs)


if __name__ == "__main__":
    # Smoke test: comportamento nos extremos e equivalência focal(gamma=0) == BCE.
    torch.manual_seed(0)

    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0

    # Logits que reproduzem o alvo com folga, e sua versão invertida.
    perfect = (target * 2.0 - 1.0) * 10.0
    inverted = -perfect

    loss_fn = build_loss("bce_dice")
    l_perfect = loss_fn(perfect, target).item()
    l_inverted = loss_fn(inverted, target).item()
    print(f"bce_dice  predição perfeita: {l_perfect:.6f}")
    print(f"bce_dice  predição invertida: {l_inverted:.6f}")
    assert l_perfect < 0.01, "perda da predição perfeita deveria ser ~0"
    assert l_inverted > 1.0, "perda da predição invertida deveria ser alta"

    # Com gamma=0 e sem alpha, a focal tem que colapsar exatamente na BCE.
    logits = torch.randn(2, 1, 32, 32)
    focal0 = FocalLoss(alpha=None, gamma=0.0)(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    print(f"focal(gamma=0)={focal0.item():.6f}  bce={bce.item():.6f}")
    assert torch.allclose(focal0, bce, atol=1e-6), "focal com gamma=0 deveria igualar a BCE"

    # gamma crescente reduz a perda dos exemplos fáceis.
    easy = (target * 2.0 - 1.0) * 3.0
    prev = float("inf")
    for gamma in (0.0, 1.0, 2.0, 5.0):
        value = FocalLoss(alpha=None, gamma=gamma)(easy, target).item()
        print(f"focal  gamma={gamma}: {value:.6f}")
        assert value < prev, "aumentar gamma deveria reduzir a perda em exemplos fáceis"
        prev = value

    print(f"pos_weight do alvo de teste: {compute_pos_weight(target):.3f}")
    print("todos os testes ok")
