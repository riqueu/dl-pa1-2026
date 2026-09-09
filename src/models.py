"""Encoders/Decoders (U-Net, ResUNet, SegNet, DeepLab/ASPP).

Este módulo implementa a arquitetura baseline das Partes 0 e 1: uma U-Net com
encoder ResNet pré-treinado em ImageNet e decoder autoral com skip connections.

Contratos de interface respeitados:
- Entrada: Tensor float32 (B, 3, H, W) normalizado em [0.0, 1.0].
- Saída: Tensor float32 (B, out_channels, H, W) com logits (sem sigmoid/softmax).

Glossário:
- encoder: metade que encolhe a imagem; enxerga cada vez mais contexto e perde detalhe.
- decoder: metade que devolve a imagem ao tamanho original, para responder pixel a pixel.
- feature map: saída de uma camada; cada canal é um padrão que a rede aprendeu a detectar.
- skip connection: atalho que leva detalhe fino do encoder direto ao decoder.
- gargalo: ponto de menor resolução, onde há mais contexto e menos detalhe.
- logits: nota crua da rede, de -inf a +inf; só vira probabilidade depois do sigmoid.
- stride: de quantos pixels o filtro anda por vez; stride 2 corta a resolução pela metade.
- pré-treinado: pesos herdados de um treino anterior (ImageNet), usados como ponto de partida.
- buffer: número que o modelo carrega junto mas nunca aprende nem ajusta.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# Constantes de normalização do pré-treino ImageNet
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Canais dos 5 níveis de features extraídos do encoder, do mais raso ao mais profundo.
# ResNet18 e ResNet34 compartilham a mesma largura por usarem BasicBlock.
_ENCODER_CHANNELS: Dict[str, Tuple[int, int, int, int, int]] = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
}


def _load_resnet(name: str, pretrained: bool) -> nn.Module:
    """Instancia uma ResNet do torchvision com ou sem pesos ImageNet.

    Args:
        name: Nome da arquitetura ('resnet18' ou 'resnet34').
        pretrained: Se True, carrega os pesos treinados em ImageNet.

    Returns:
        Instância da ResNet correspondente.

    Raises:
        ValueError: Se o encoder solicitado não for suportado.
    """
    if name not in _ENCODER_CHANNELS:
        raise ValueError(
            f"Encoder '{name}' não suportado. Disponíveis: {sorted(_ENCODER_CHANNELS)}."
        )

    factory = getattr(torchvision.models, name)

    # A API de pesos do torchvision mudou de `pretrained=` para `weights=`.
    try:
        weights_enum = torchvision.models.get_model_weights(name).DEFAULT if pretrained else None
        return factory(weights=weights_enum)
    except (AttributeError, TypeError):
        return factory(pretrained=pretrained)


class ResNetEncoder(nn.Module):
    """Encoder ResNet que expõe os 5 níveis de resolução usados pelas skip connections.

    As tomadas de features seguem a estrutura da ResNet do torchvision:

    | Nível | Origem                  | Resolução | Canais (R18/R34) |
    |-------|-------------------------|-----------|------------------|
    | 0     | relu(bn1(conv1))        | H/2       | 64               |
    | 1     | layer1 (após maxpool)   | H/4       | 64               |
    | 2     | layer2                  | H/8       | 128              |
    | 3     | layer3                  | H/16      | 256              |
    | 4     | layer4 (gargalo)        | H/32      | 512              |
    """

    def __init__(self, name: str = "resnet34", pretrained: bool = True) -> None:
        """Inicializa o encoder.

        Args:
            name: Arquitetura base ('resnet18' ou 'resnet34').
            pretrained: Se True, aproveita os pesos ImageNet.
        """
        super().__init__()
        net = _load_resnet(name, pretrained)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.out_channels: Tuple[int, ...] = _ENCODER_CHANNELS[name]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extrai as features multi-escala.

        Args:
            x: Tensor (B, 3, H, W) já normalizado.

        Returns:
            Lista com 5 tensores, do mais raso (H/2) ao gargalo (H/32).
        """
        f0 = self.stem(x)
        f1 = self.layer1(self.pool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]


class Upsample(nn.Module):
    """Dobra a resolução espacial pelo mecanismo escolhido.

    Mecanismos suportados:
    - 'transpose': convolução transposta 2x2 com stride 2, isto é, ampliação com
      pesos que a rede aprende, em vez de uma regra fixa de interpolação.
    - 'bilinear': interpolação bilinear (sem parâmetros).
    - 'nearest': interpolação por vizinho mais próximo (sem parâmetros).

    Pool indices: anotar em que posição estava o maior valor de cada janela ao
    encolher, para devolver o valor exatamente ali ao ampliar de volta.

    O mecanismo de pool indices do SegNet não é oferecido aqui porque a ResNet
    reduz resolução por convolução com stride, não por max pooling, e portanto não
    produz os índices necessários. Comparar pool indices no Eixo 1 da Parte 3 exige
    um encoder do tipo VGG/SegNet, a ser adicionado quando aquela ablação começar.
    """

    def __init__(self, channels: int, mode: str = "transpose") -> None:
        """Inicializa o módulo de subida.

        Args:
            channels: Número de canais de entrada (mantido na saída).
            mode: 'transpose', 'bilinear' ou 'nearest'.

        Raises:
            ValueError: Se o modo não for reconhecido.
            NotImplementedError: Se for solicitado 'unpool'.
        """
        super().__init__()
        self.mode = mode

        if mode == "transpose":
            self.up: nn.Module = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
            
        elif mode in ("bilinear", "nearest"):
            self.up = nn.Identity()
            
        elif mode == "unpool":
            # Método de Pool não implementado...
            raise NotImplementedError(
                "up_mode='unpool' exige um encoder que reduza resolução por max pooling "
                "(VGG/SegNet) para fornecer os pool indices. A ResNet usa stride."
            )
        else:
            raise ValueError(
                f"up_mode '{mode}' desconhecido. Use 'transpose', 'bilinear' ou 'nearest'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            return self.up(x)
        align = False if self.mode == "bilinear" else None
        return F.interpolate(x, scale_factor=2.0, mode=self.mode, align_corners=align)


class DecoderBlock(nn.Module):
    """Um estágio de subida do decoder: upsample, concatenação da skip e duas convoluções."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_mode: str = "transpose",
    ) -> None:
        """Inicializa o bloco.

        Args:
            in_channels: Canais vindos do nível inferior do decoder.
            skip_channels: Canais da skip connection (0 quando não há skip).
            out_channels: Canais produzidos pelo bloco.
            up_mode: Mecanismo de recuperação de resolução.
        """
        super().__init__()
        self.up = Upsample(in_channels, mode=up_mode)

        conv_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sobe um nível de resolução e funde a skip correspondente.

        Args:
            x: Tensor vindo do nível inferior do decoder.
            skip: Feature map do encoder na resolução alvo, ou None.

        Returns:
            Tensor com o dobro da resolução espacial de `x`.
        """
        x = self.up(x)

        if skip is not None:
            # Realinha caso a entrada não seja múltipla de 32 e a subida gere off-by-one.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """U-Net com encoder ResNet pré-treinado e decoder autoral.

    A normalização ImageNet é aplicada internamente, como primeira operação do
    forward, de modo que a rede aceita diretamente as imagens em [0, 1] entregues
    pelos datasets do projeto.
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        out_channels: int = 1,
        up_mode: str = "transpose",
        use_skips: bool = True,
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    ) -> None:
        """Inicializa a U-Net.

        Args:
            encoder: Backbone ('resnet18' ou 'resnet34').
            out_channels: Canais de saída. 1 na baseline binária; 3 na Trilha A;
                D na Trilha B; 3 na Trilha C.
            up_mode: Mecanismo de subida ('transpose', 'bilinear' ou 'nearest').
            use_skips: Se False, desliga todas as skip connections (ablação do Eixo 1).
            pretrained: Se True, inicializa o encoder com pesos ImageNet.
            decoder_channels: Canais de saída de cada um dos 5 estágios do decoder.

        Raises:
            ValueError: Se `decoder_channels` não tiver exatamente 5 elementos.
        """
        super().__init__()

        if len(decoder_channels) != 5:
            raise ValueError(
                f"decoder_channels precisa ter 5 elementos, recebeu {len(decoder_channels)}."
            )

        self.encoder_name = encoder
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.use_skips = use_skips

        self.encoder = ResNetEncoder(encoder, pretrained=pretrained)
        enc_ch = self.encoder.out_channels  # (H/2, H/4, H/8, H/16, H/32)

        # Canais de skip disponíveis em cada estágio de subida, do gargalo para cima.
        # O último estágio (H/2 -> H) não tem skip: o stem já é o nível mais raso.
        skip_ch = [enc_ch[3], enc_ch[2], enc_ch[1], enc_ch[0], 0]
        if not use_skips:
            skip_ch = [0, 0, 0, 0, 0]

        blocks: List[nn.Module] = []
        in_ch = enc_ch[4]
        for stage, out_ch in enumerate(decoder_channels):
            blocks.append(DecoderBlock(in_ch, skip_ch[stage], out_ch, up_mode=up_mode))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)

        self.head = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

        # Buffers não-persistentes: viajam com .to(device) e ficam fora do state_dict.
        self.register_buffer(
            "norm_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "norm_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a rede.

        Args:
            x: Tensor float32 (B, 3, H, W) em [0.0, 1.0], sem normalização prévia.

        Returns:
            Tensor float32 (B, out_channels, H, W) com logits crus.
        """
        input_size = x.shape[-2:]

        # Trava 2: a normalização do pré-treino é responsabilidade do modelo.
        x = (x - self.norm_mean) / self.norm_std

        features = self.encoder(x)
        skips = [features[3], features[2], features[1], features[0], None]

        out = features[4]
        for stage, block in enumerate(self.decoder):
            skip = skips[stage] if self.use_skips else None
            out = block(out, skip)

        # Garante saída na resolução exata da entrada, mesmo para H ou W não múltiplos de 32.
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return self.head(out)


def build_model(
    encoder: str = "resnet34",
    out_channels: int = 1,
    up_mode: str = "transpose",
    use_skips: bool = True,
    pretrained: bool = True,
) -> UNet:
    """Constrói o modelo do projeto.

    Chamado sem argumentos por `evaluate.py` e por `notebooks/inferencia.ipynb`,
    e portanto os defaults precisam reproduzir exatamente a arquitetura da baseline
    treinada — `load_state_dict` é estrito e falha com qualquer divergência.

    Args:
        encoder: Backbone ('resnet18' ou 'resnet34').
        out_channels: Canais de saída da cabeça.
        up_mode: Mecanismo de subida do decoder.
        use_skips: Liga ou desliga as skip connections.
        pretrained: Inicialização com pesos ImageNet.

    Returns:
        Instância de UNet pronta para treino ou inferência.
    """
    return UNet(
        encoder=encoder,
        out_channels=out_channels,
        up_mode=up_mode,
        use_skips=use_skips,
        pretrained=pretrained,
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Conta os parâmetros do modelo.

    Args:
        model: Módulo a inspecionar.
        trainable_only: Se True, ignora parâmetros congelados.

    Returns:
        Número de parâmetros.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


if __name__ == "__main__":
    # Smoke test: shapes de entrada e saída, e backward funcional.
    torch.manual_seed(0)

    model = build_model(pretrained=False)
    print(f"Parâmetros treináveis: {count_parameters(model):,}")

    for h, w in [(128, 128), (256, 256), (200, 200)]:
        x = torch.rand(2, 3, h, w)
        y = model(x)
        assert y.shape == (2, 1, h, w), f"esperado (2, 1, {h}, {w}), obtido {tuple(y.shape)}"
        print(f"entrada (2, 3, {h}, {w}) -> saída {tuple(y.shape)}  ok")

    loss = model(torch.rand(2, 3, 128, 128)).mean()
    loss.backward()
    print("backward ok")
