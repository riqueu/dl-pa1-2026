"""CE balanceada, Focal Loss, L1/L2.

Perdas da baseline binária (Partes 0 e 1), da cabeça de fronteiras (Parte 2)
e da ablação do Eixo 2 (Parte 3).

Todas as perdas compartilham a assinatura ``(logits, target)``. As binárias
recebem logits ``(B, 1, H, W)`` e alvo float; as multiclasse recebem logits
``(B, C, H, W)`` e alvo int64 ``(B, H, W)``.

Glossário:
- logits: nota crua da rede, de -inf a +inf; só vira probabilidade depois do sigmoid.
- BCE: olha cada pixel isolado e castiga conforme a distância entre nota e resposta certa.
- Dice: mede o quanto a área prevista e a área real se sobrepõem; ignora acerto no fundo.
- Soft Dice: mesma conta, usando a probabilidade em vez de sim/não, para poder treinar.
- desbalanceamento: quando uma classe ocupa quase toda a imagem e a outra quase nada.
- pos_weight: multiplicador que aumenta o castigo por errar a classe rara.
- alpha: na focal, quanto peso dar à classe rara.
- gamma: na focal, o quanto ignorar os exemplos que a rede já acerta com folga.
"""

from typing import Any, Dict, Optional, Sequence, Union

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


class WeightedCrossEntropyLoss(nn.Module):
    """Entropia cruzada multiclasse com peso independente por classe.

    Na Trilha A, a fronteira ocupa uma fração pequena da imagem. Os pesos
    compensam essa frequência sem modificar o alvo categórico.
    """

    def __init__(self, class_weights: Optional[Sequence[float]] = None) -> None:
        """Inicializa a perda.

        Args:
            class_weights: Pesos na ordem fundo, interior e fronteira. ``None``
                reproduz a entropia cruzada multiclasse comum.
        """
        super().__init__()
        weights = None
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.ndim != 1 or weights.numel() == 0:
                raise ValueError("class_weights deve ser uma sequência unidimensional não vazia.")
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Todos os pesos de classe devem ser positivos e finitos.")
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcula a CE média sobre todos os pixels."""
        return F.cross_entropy(logits, target.long(), weight=self.class_weights)


class MulticlassFocalLoss(nn.Module):
    """Focal loss multiclasse para a cabeça fundo/interior/fronteira.

    Para cada pixel, multiplica a CE da classe verdadeira por
    ``(1 - p_t) ** gamma``. ``gamma=0`` reproduz exatamente a CE, inclusive
    quando os pesos ``alpha`` estão presentes.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Sequence[float]] = None,
    ) -> None:
        """Inicializa a perda.

        Args:
            gamma: Expoente não negativo do fator focal.
            alpha: Pesos por classe, na ordem dos canais, ou ``None``.
        """
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma deve ser maior ou igual a zero.")
        self.gamma = gamma

        alpha_tensor = None
        if alpha is not None:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.ndim != 1 or alpha_tensor.numel() == 0:
                raise ValueError("alpha deve ser uma sequência unidimensional não vazia.")
            if not torch.isfinite(alpha_tensor).all() or (alpha_tensor <= 0).any():
                raise ValueError("Todos os valores de alpha devem ser positivos e finitos.")
        self.register_buffer("alpha", alpha_tensor)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcula a focal multiclasse média de forma numericamente estável."""
        if logits.ndim < 3:
            raise ValueError(f"logits multiclasse deve ter ao menos 3 eixos; recebeu {logits.shape}.")
        if self.alpha is not None and self.alpha.numel() != logits.shape[1]:
            raise ValueError(
                f"alpha tem {self.alpha.numel()} pesos, mas logits tem {logits.shape[1]} classes."
            )

        log_probs = F.log_softmax(logits, dim=1)
        target = target.long()
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).pow(self.gamma)) * log_pt

        if self.alpha is not None:
            pixel_weights = self.alpha[target]
            loss = loss * pixel_weights
            return loss.sum() / pixel_weights.sum()

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


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int = 3,
) -> torch.Tensor:
    """Calcula pesos por frequência inversa para alvos categóricos.

    A fórmula ``N / (C * N_c)`` dá peso maior às classes raras e mantém a
    contribuição média ponderada em torno de um. Deve ser aplicada apenas aos
    alvos do split de treino.

    Args:
        targets: Tensor inteiro contendo os IDs de classe.
        num_classes: Número total de classes esperado.

    Returns:
        Tensor float32 ``(num_classes,)`` com os pesos.

    Raises:
        ValueError: Se houver IDs fora do intervalo ou alguma classe estiver
            ausente da amostra usada para estimar os pesos.
    """
    if num_classes <= 0:
        raise ValueError("num_classes deve ser positivo.")

    flat = targets.detach().long().reshape(-1).cpu()
    if flat.numel() == 0:
        raise ValueError("targets não pode ser vazio.")
    if flat.min().item() < 0 or flat.max().item() >= num_classes:
        raise ValueError(f"targets deve conter apenas IDs entre 0 e {num_classes - 1}.")

    counts = torch.bincount(flat, minlength=num_classes).to(torch.float64)
    if (counts == 0).any():
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"Não é possível pesar classes ausentes: {missing}.")

    weights = counts.sum() / (num_classes * counts)
    return weights.to(torch.float32)


def build_loss(name: str = "bce_dice", **kwargs: Any) -> nn.Module:
    """Constrói a perda pelo nome, para seleção via linha de comando.

    Args:
        name: Nome de uma perda binária ou multiclasse registrada.
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
        "weighted_ce_3c": WeightedCrossEntropyLoss,
        "multiclass_focal": MulticlassFocalLoss,
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

    # Perdas da Parte 2: shapes, backward e equivalência focal(gamma=0) == CE.
    target_3c = torch.randint(0, 3, (2, 32, 32))
    logits_3c = torch.randn(2, 3, 32, 32, requires_grad=True)
    class_weights = compute_class_weights(target_3c)
    ce_3c = WeightedCrossEntropyLoss(class_weights)(logits_3c, target_3c)
    focal_3c = MulticlassFocalLoss(gamma=0.0, alpha=class_weights)(logits_3c, target_3c)
    assert torch.allclose(focal_3c, ce_3c, atol=1e-6)
    ce_3c.backward()
    assert logits_3c.grad is not None and torch.isfinite(logits_3c.grad).all()
    print(f"pesos 3 classes: {[round(v, 3) for v in class_weights.tolist()]}")
    print(f"CE 3 classes: {ce_3c.item():.6f} | focal(gamma=0): {focal_3c.item():.6f}")
    print("todos os testes ok")
