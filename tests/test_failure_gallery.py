"""Testes da análise morfológica usada na Parte 5."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from scripts.run_failure_gallery import compute_dataset_nuclei_distribution


class FailureGalleryTests(unittest.TestCase):
    def test_diametro_e_medido_apos_resize_para_entrada_da_rede(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            masks_dir = os.path.join(root, "imagem_1", "masks")
            os.makedirs(masks_dir)
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[100:200, 100:200] = 255
            cv2.imwrite(os.path.join(masks_dir, "nucleo.png"), mask)

            stats = compute_dataset_nuclei_distribution(root, target_size=(256, 256))

        self.assertEqual(stats["analysis_resolution"], [256, 256])
        self.assertEqual(stats["source_nuclei"], 1)
        self.assertEqual(stats["total_nuclei"], 1)
        self.assertEqual(float(stats["areas"][0]), 2500.0)


if __name__ == "__main__":
    unittest.main()
