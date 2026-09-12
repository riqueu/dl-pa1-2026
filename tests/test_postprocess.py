"""Testes unitários para o pós-processamento e decodificação Watershed (Parte 2 e Parte 5)."""

import unittest
import numpy as np

from src.postprocess import (
    watershed_instance_segmentation,
    watershed_instance_segmentation_batch,
)


class TestWatershedPostprocess(unittest.TestCase):
    """Testa a decodificação topográfica Watershed e a correção morfológica de sementes."""

    def test_watershed_separates_touching_nuclei(self) -> None:
        """Testa se dois núcleos tocantes com marcadores distintos são separados."""
        prob = np.zeros((3, 30, 30), dtype=np.float32)
        # Background
        prob[0, :, :] = 1.0
        # Nucleus 1: interior em (10, 8), boundary ao redor
        prob[1, 7:13, 6:11] = 0.9
        # Nucleus 2: interior em (10, 18), boundary ao redor
        prob[1, 7:13, 16:21] = 0.9
        # Fronteira compartilhada no meio
        prob[2, 7:13, 12:15] = 0.95
        # Foreground comum
        prob[0, 6:14, 5:22] = 0.05

        labels = watershed_instance_segmentation(prob, interior_threshold=0.5, foreground_threshold=0.5)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]
        self.assertEqual(len(unique_labels), 2, "Watershed deve separar os dois núcleos.")

    def test_seed_closing_fuses_fragmented_internal_markers(self) -> None:
        """Testa se seed_closing_radius consolida múltiplos marcadores dentro do mesmo núcleo grande."""
        prob = np.zeros((3, 40, 40), dtype=np.float32)
        prob[0, :, :] = 1.0

        # Célula grande de 20x20 pixels (toda é foreground)
        prob[0, 10:30, 10:30] = 0.05
        prob[1, 10:30, 10:30] = 0.45

        # Dois marcadores fortes de interior (>= 0.5) separados por variação de cromatina interna
        prob[1, 12:18, 12:28] = 0.85
        prob[1, 22:28, 12:28] = 0.85

        # Fronteira externa da célula
        prob[2, 9:11, 9:31] = 0.9
        prob[2, 29:31, 9:31] = 0.9
        prob[2, 9:31, 9:11] = 0.9
        prob[2, 9:31, 29:31] = 0.9

        # Sem closing morfológico: 2 marcadores viram 2 instâncias fragmentadas
        labels_split = watershed_instance_segmentation(
            prob, interior_threshold=0.5, foreground_threshold=0.4, seed_closing_radius=0
        )
        n_split = len([x for x in np.unique(labels_split) if x != 0])
        self.assertEqual(n_split, 2, "Sem closing, devem surgir 2 instâncias fragmentadas.")

        # Com closing morfológico (radius=3): sementes fundem-se em 1 marcador único
        labels_fused = watershed_instance_segmentation(
            prob, interior_threshold=0.5, foreground_threshold=0.4, seed_closing_radius=3
        )
        n_fused = len([x for x in np.unique(labels_fused) if x != 0])
        self.assertEqual(n_fused, 1, "Com seed_closing_radius=3, os marcadores internos devem se fundir.")

    def test_seed_closing_validation_errors(self) -> None:
        """Valida que valores negativos de seed_closing_radius disparam erro."""
        prob = np.zeros((3, 20, 20), dtype=np.float32)
        with self.assertRaises(ValueError):
            watershed_instance_segmentation(prob, seed_closing_radius=-1)


if __name__ == "__main__":
    unittest.main()
