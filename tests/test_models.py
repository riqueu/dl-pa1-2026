"""Testes estruturais das arquiteturas da ablação da Parte 3."""

import tempfile
import unittest

import torch

from src.models import build_model
from src.utils import resnet34_receptive_field_summary


class ModelArchitectureTests(unittest.TestCase):
    def test_defaults_preservam_unet_baseline(self) -> None:
        default_model = build_model(pretrained=False)
        explicit_model = build_model(
            encoder="resnet34",
            out_channels=1,
            up_mode="transpose",
            use_skips=True,
            pretrained=False,
            decoder_type="unet",
            output_stride=32,
        )
        self.assertEqual(list(default_model.state_dict()), list(explicit_model.state_dict()))

    def test_deeplab_preserva_resolucao_de_saida(self) -> None:
        model = build_model(
            out_channels=3,
            pretrained=False,
            decoder_type="aspp",
        ).eval()
        for height, width in ((128, 128), (97, 113)):
            with self.subTest(shape=(height, width)), torch.no_grad():
                output = model(torch.rand(1, 3, height, width))
                self.assertEqual(tuple(output.shape), (1, 3, height, width))

    def test_encoder_deeplab_tem_output_stride_16(self) -> None:
        model = build_model(pretrained=False, decoder_type="aspp").eval()
        with torch.no_grad():
            features = model.encoder(torch.rand(1, 3, 128, 128))
        self.assertEqual(tuple(features[-2].shape[-2:]), (8, 8))
        self.assertEqual(tuple(features[-1].shape[-2:]), (8, 8))

    def test_checkpoint_deeplab_recarrega_estritamente(self) -> None:
        args = {
            "encoder": "resnet34",
            "out_channels": 3,
            "up_mode": "transpose",
            "use_skips": True,
            "pretrained": False,
            "decoder_type": "aspp",
            "output_stride": 16,
            "aspp_rates": [6, 12, 18],
        }
        original = build_model(**args)
        with tempfile.NamedTemporaryFile(suffix=".pth") as file:
            torch.save({"model_state_dict": original.state_dict(), "model_args": args}, file.name)
            state = torch.load(file.name, map_location="cpu", weights_only=True)
        restored = build_model(**state["model_args"])
        restored.load_state_dict(state["model_state_dict"], strict=True)

    def test_campo_receptivo_reflete_dilatacao(self) -> None:
        standard = resnet34_receptive_field_summary(output_stride=32)
        os16_without_atrous = resnet34_receptive_field_summary(
            output_stride=16,
            dilate_layer4=False,
        )
        atrous = resnet34_receptive_field_summary(output_stride=16)
        self.assertEqual(standard["encoder_jump"], 32)
        self.assertEqual(os16_without_atrous["encoder_jump"], 16)
        self.assertEqual(atrous["encoder_jump"], 16)
        self.assertGreater(
            atrous["encoder_receptive_field"],
            os16_without_atrous["encoder_receptive_field"],
        )
        self.assertGreater(
            atrous["aspp_branch_receptive_fields"]["conv_3x3_rate_18"],
            atrous["encoder_receptive_field"],
        )


if __name__ == "__main__":
    unittest.main()
