"""Testes unitários automatizados para o módulo src/mosaic.py (Parte 4)."""

import numpy as np
import torch
import torch.nn as nn

from src.mosaic import (
    extract_tiles,
    detect_split_nuclei,
    predict_tiled_naive,
)


def test_extract_tiles_geometry() -> None:
    """Valida a geometria e cobertura completa das janelas deslizantes."""
    canvas = np.zeros((512, 512, 3), dtype=np.float32)
    tiles = extract_tiles(canvas, tile_size=256, stride=128)

    # Com 512x512, patch 256 e stride 128, a grade é 3x3 = 9 janelas
    assert len(tiles) == 9

    coverage_map = np.zeros((512, 512), dtype=np.int32)
    for t in tiles:
        ymin, xmin, ymax, xmax = t["box"]
        assert ymax - ymin == 256
        assert xmax - xmin == 256
        assert t["tile"].shape == (256, 256, 3)
        coverage_map[ymin:ymax, xmin:xmax] += 1

    # Toda a área deve estar coberta por pelo menos 1 tile
    assert np.all(coverage_map >= 1)
    # Zonas de overlap devem ter até 4 tiles sobrepostos
    assert coverage_map.max() == 4


def test_detect_split_nuclei_synthetic() -> None:
    """Verifica detecção analítica de núcleos que cruzam linhas de corte."""
    gt = np.zeros((512, 512), dtype=np.int64)

    # 1. Objeto interno no quadrante (não cruza bordas)
    gt[50:70, 50:70] = 1

    # 2. Objeto que cruza x = 128 (borda de janela deslizante)
    gt[50:70, 120:140] = 2

    # 3. Objeto que cruza y = 256 (borda central do mosaico)
    gt[250:270, 50:70] = 3

    tile_boxes = [
        (0, 0, 256, 256),
        (0, 128, 256, 384),
        (0, 256, 256, 512),
        (128, 0, 384, 256),
        (128, 128, 384, 384),
        (128, 256, 384, 512),
        (256, 0, 512, 256),
        (256, 128, 512, 384),
        (256, 256, 512, 512),
    ]

    analysis = detect_split_nuclei(gt, tile_boxes)

    assert analysis["total_gt_nuclei"] == 3
    assert 1 not in analysis["boundary_gt_ids"]
    assert 2 in analysis["boundary_gt_ids"]
    assert 3 in analysis["boundary_gt_ids"]
    assert analysis["num_boundary_nuclei"] == 2


class MockModel(nn.Module):
    """Mock da U-Net que retorna 3 canais de logits simulando objetos."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Canal 0: fundo alto por padrão
        out = torch.full((b, 3, h, w), -2.0, device=x.device)
        out[:, 0] = 5.0
        # Cria um núcleo no centro do patch
        cy, cx = h // 2, w // 2
        out[:, 0, cy-15:cy+15, cx-15:cx+15] = -5.0
        out[:, 1, cy-10:cy+10, cx-10:cx+10] = 6.0  # interior
        out[:, 2, cy-15:cy+15, cx-15:cx+15] = 4.0  # fronteira
        return out


def test_predict_tiled_naive_modes() -> None:
    """Testa a execução do pipeline de Tiling Ingênuo em ambos os modos."""
    mock_model = MockModel()
    mock_model.eval()

    img = np.random.rand(512, 512, 3).astype(np.float32)

    # Modo center_crop
    res_crop = predict_tiled_naive(img, model=mock_model, tile_size=256, stride=128, mode="center_crop")
    assert "naive_instance_mask" in res_crop
    assert res_crop["naive_instance_mask"].shape == (512, 512)
    assert len(res_crop["tiles_preds"]) == 9
    assert len(res_crop["tile_boxes"]) == 9

    # Modo direct_stamp
    res_stamp = predict_tiled_naive(img, model=mock_model, tile_size=256, stride=128, mode="direct_stamp")
    assert "naive_instance_mask" in res_stamp
    assert res_stamp["naive_instance_mask"].shape == (512, 512)
    assert res_stamp["num_instances"] >= res_crop["num_instances"]


if __name__ == "__main__":
    print("Rodando test_extract_tiles_geometry...")
    test_extract_tiles_geometry()
    print("Rodando test_detect_split_nuclei_synthetic...")
    test_detect_split_nuclei_synthetic()
    print("Rodando test_predict_tiled_naive_modes...")
    test_predict_tiled_naive_modes()
    print("Todos os testes unitários da Parte 4 passaram com sucesso!")
