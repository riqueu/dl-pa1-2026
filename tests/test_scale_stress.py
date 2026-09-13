"""Testes deterministas do protocolo de mudanca de escala da Parte 6."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from src.scale_stress import (
    add_native_relative_metrics,
    aggregate_instance_metrics,
    decode_scaled_probabilities,
    parse_scale_factors,
    predict_scaled_probabilities,
    resolve_min_area,
    resize_image_batch,
    restore_instance_mask,
)


class _ThreeClassModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        return torch.zeros(batch, 3, height, width, device=x.device)


class ScaleStressTests(unittest.TestCase):
    def test_parseia_escalas_e_rejeita_invalidas(self) -> None:
        self.assertEqual(parse_scale_factors("0.5, 1, 2"), (0.5, 1.0, 2.0))
        for spec in ("", "0,1", "1,1", "x,2"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_scale_factors(spec)

    def test_redimensionamento_da_imagem_preserva_contrato(self) -> None:
        images = torch.rand(2, 3, 256, 256)
        self.assertEqual(tuple(resize_image_batch(images, 0.5).shape), (2, 3, 128, 128))
        self.assertEqual(tuple(resize_image_batch(images, 2.0).shape), (2, 3, 512, 512))
        self.assertIs(resize_image_batch(images, 1.0), images)

    def test_area_fixa_e_area_fisica_controlada(self) -> None:
        self.assertEqual(resolve_min_area(10, 0.5, "fixed"), 10)
        self.assertEqual(resolve_min_area(10, 0.5, "scaled"), 2)
        self.assertEqual(resolve_min_area(10, 2.0, "scaled"), 40)
        self.assertEqual(resolve_min_area(0, 2.0, "scaled"), 0)

    def test_restaura_ids_por_vizinho_mais_proximo(self) -> None:
        labels = np.array([[0, 7], [11, 11]], dtype=np.int64)
        restored = restore_instance_mask(labels, (4, 4))
        self.assertEqual(restored.shape, (4, 4))
        self.assertEqual(restored.dtype, np.int64)
        self.assertEqual(set(np.unique(restored)), {0, 7, 11})

    def test_decodificacao_retorna_resolucao_nativa(self) -> None:
        probabilities = np.zeros((1, 3, 8, 8), dtype=np.float32)
        probabilities[:, 0] = 1.0
        probabilities[:, 0, 2:6, 2:6] = 0.0
        probabilities[:, 1, 2:6, 2:6] = 1.0
        predictions = decode_scaled_probabilities(
            probabilities,
            native_size=(16, 16),
            scale=0.5,
            base_min_area=0,
        )
        self.assertEqual(predictions[0].shape, (16, 16))
        self.assertEqual(set(np.unique(predictions[0])), {0, 1})

    def test_forward_escalado_exige_tres_classes(self) -> None:
        images = torch.rand(1, 3, 16, 16)
        probabilities = predict_scaled_probabilities(_ThreeClassModel(), images, 0.5)
        self.assertEqual(probabilities.shape, (1, 3, 8, 8))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_agrega_metricas_e_calcula_retencao(self) -> None:
        results = [
            {"mAP": 0.4, "count_error": 2, "AP_per_iou": {0.5: 0.8, 0.75: 0.5}},
            {"mAP": 0.6, "count_error": 4, "AP_per_iou": {0.5: 1.0, 0.75: 0.7}},
        ]
        aggregate = aggregate_instance_metrics(results)
        self.assertAlmostEqual(aggregate["mean_mAP"], 0.5)
        self.assertAlmostEqual(aggregate["mean_AP50"], 0.9)
        self.assertAlmostEqual(aggregate["mean_AP75"], 0.6)
        self.assertAlmostEqual(aggregate["mean_count_error"], 3.0)

        relative = add_native_relative_metrics(
            {0.5: {"mean_mAP": 0.25}, 1.0: {"mean_mAP": 0.5}, 2.0: {"mean_mAP": 0.4}}
        )
        self.assertAlmostEqual(relative[0.5]["mAP_retention_vs_1x"], 0.5)
        self.assertAlmostEqual(relative[2.0]["mAP_degradation_percent_vs_1x"], 20.0)


if __name__ == "__main__":
    unittest.main()
