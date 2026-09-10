"""Componentes conexos, Watershed / Centros+Offsets / Embeddings.

Decodificação da saída densa da rede em objetos numerados. A Parte 1 usa o
método ingênuo (limiar + componentes conexos); as trilhas da Parte 2 entram aqui
depois, sem alterar a assinatura de saída.

Contrato de saída: array NumPy int64 (H, W) com 0 = fundo e 1..K = IDs de
instâncias, consecutivos.

Glossário:
- componentes conexos: grupos de pixels acesos que se tocam; cada grupo vira um objeto.
- conectividade: define o que é "se tocar". 4 = só pelos lados; 8 = lados e diagonais.
- limiar: nota de corte que transforma probabilidade em decisão sim/não.
- instância: um objeto individual, com um número próprio que o distingue dos vizinhos.

Regras de decodificação escolhidas (a Parte 1 exige que sejam explícitas):
- Conectividade 4 por padrão, a mais conservadora: núcleos que só se encostam
  na diagonal continuam separados.
- Sem filtro de área por padrão (`min_size=0`), para que a baseline seja de fato
  ingênua. O filtro existe para ser calibrado na validação, nunca no teste.
"""

from typing import List, Union

import numpy as np
import torch
from scipy import ndimage
from skimage.segmentation import watershed


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy_2d(x: ArrayLike) -> np.ndarray:
    """Converte a entrada em um array 2D (H, W), aceitando tensores e eixos unitários.

    Args:
        x: Array ou tensor de shape (H, W), (1, H, W) ou (1, 1, H, W).

    Returns:
        Array NumPy 2D.

    Raises:
        ValueError: Se sobrar mais de uma imagem após remover os eixos unitários.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    # Remove eixos de tamanho 1 à esquerda (lote e canal).
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 2:
        raise ValueError(
            f"Esperado uma única imagem (H, W); recebido shape {x.shape}. "
            "Para lotes, use naive_connected_components_batch."
        )

    return x


def _to_numpy_three_channels(x: ArrayLike) -> np.ndarray:
    """Converte uma predição individual para o formato ``(3, H, W)``."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(
            f"Esperado shape (3, H, W) ou (1, 3, H, W); recebido {x.shape}."
        )
    if not np.isfinite(x).all():
        raise ValueError("A predição contém valores não finitos.")
    return x.astype(np.float32, copy=False)


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """Renumera os rótulos para 1..K sem buracos, preservando o fundo em 0.

    Args:
        labels: Array (H, W) com rótulos inteiros possivelmente não consecutivos.

    Returns:
        Array (H, W) int64 com rótulos consecutivos.
    """
    present = np.unique(labels)
    present = present[present != 0]

    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    lookup[present] = np.arange(1, len(present) + 1, dtype=np.int64)

    return lookup[labels]


def naive_connected_components(
    x: ArrayLike,
    threshold: float = 0.5,
    from_logits: bool = True,
    connectivity: int = 1,
    min_size: int = 0,
) -> np.ndarray:
    """Extrai instâncias por limiar seguido de componentes conexos (Parte 1, item 2).

    Este é o método ingênuo, e falha por construção quando dois núcleos se tocam:
    para ele, uma mancha conexa é um objeto. Essa falha é a evidência que a Parte 1
    pede para quantificar.

    Args:
        x: Saída da rede para uma imagem, shape (H, W), (1, H, W) ou (1, 1, H, W).
        threshold: Nota de corte aplicada à probabilidade.
        from_logits: Se True, aplica sigmoid antes do limiar. Use False quando a
            entrada já for probabilidade.
        connectivity: 1 para vizinhança de 4, 2 para vizinhança de 8.
        min_size: Descarta componentes com menos pixels que este valor. 0 desliga.

    Returns:
        Array NumPy int64 (H, W) com 0 = fundo e 1..K = IDs consecutivos.

    Raises:
        ValueError: Se `connectivity` não for 1 ou 2.
    """
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity deve ser 1 (4-vizinhos) ou 2 (8-vizinhos); recebido {connectivity}.")

    arr = _to_numpy_2d(x).astype(np.float32)

    if from_logits:
        # Sigmoid estável: evita overflow do exp para logits muito negativos.
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))

    binary = arr >= threshold

    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, n_labels = ndimage.label(binary, structure=structure)

    if n_labels == 0:
        return np.zeros_like(labels, dtype=np.int64)

    if min_size > 0:
        # bincount conta pixels por rótulo de uma vez; índice 0 é o fundo.
        counts = np.bincount(labels.ravel())
        too_small = np.flatnonzero(counts < min_size)
        too_small = too_small[too_small != 0]
        if too_small.size > 0:
            labels[np.isin(labels, too_small)] = 0

    return _relabel_consecutive(labels.astype(np.int64))


def naive_connected_components_batch(
    x: ArrayLike,
    threshold: float = 0.5,
    from_logits: bool = True,
    connectivity: int = 1,
    min_size: int = 0,
) -> List[np.ndarray]:
    """Aplica o método ingênuo a um lote inteiro.

    Args:
        x: Saída da rede com shape (B, 1, H, W) ou (B, H, W).
        threshold: Nota de corte aplicada à probabilidade.
        from_logits: Se True, aplica sigmoid antes do limiar.
        connectivity: 1 para vizinhança de 4, 2 para vizinhança de 8.
        min_size: Descarta componentes menores que este valor.

    Returns:
        Lista com B arrays (H, W) int64.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)
    if x.ndim == 4:
        x = x[:, 0]

    return [
        naive_connected_components(
            img,
            threshold=threshold,
            from_logits=from_logits,
            connectivity=connectivity,
            min_size=min_size,
        )
        for img in x
    ]


def watershed_instance_segmentation(
    prob: ArrayLike,
    interior_threshold: float = 0.50,
    foreground_threshold: float = 0.50,
    min_area: int = 10,
    connectivity: int = 1,
) -> np.ndarray:
    """Decodifica fundo/interior/fronteira em instâncias via watershed.

    Os componentes conexos do interior são os marcadores. A soma das
    probabilidades de interior e fronteira delimita o foreground, enquanto a
    probabilidade de fronteira forma a superfície topográfica: regiões de
    fronteira são cristas que separam a inundação iniciada em cada marcador.

    Args:
        prob: Probabilidades ``(3, H, W)`` na ordem fundo, interior e fronteira.
        interior_threshold: Limiar para formar os marcadores de interior.
        foreground_threshold: Limiar de ``P(interior) + P(fronteira)``.
        min_area: Remove instâncias previstas com menos pixels. Zero desliga.
        connectivity: 1 para vizinhança de 4; 2 para vizinhança de 8.

    Returns:
        Array NumPy int64 ``(H, W)``, com 0 = fundo e IDs consecutivos.
    """
    if not 0.0 <= interior_threshold <= 1.0:
        raise ValueError("interior_threshold deve estar entre 0 e 1.")
    if not 0.0 <= foreground_threshold <= 1.0:
        raise ValueError("foreground_threshold deve estar entre 0 e 1.")
    if min_area < 0:
        raise ValueError("min_area deve ser maior ou igual a zero.")
    if connectivity not in (1, 2):
        raise ValueError("connectivity deve ser 1 (4-vizinhos) ou 2 (8-vizinhos).")

    probabilities = _to_numpy_three_channels(prob)
    p_interior = probabilities[1]
    p_boundary = probabilities[2]

    foreground = (p_interior + p_boundary) >= foreground_threshold
    seeds = (p_interior >= interior_threshold) & foreground
    structure = ndimage.generate_binary_structure(2, connectivity)
    markers, n_markers = ndimage.label(seeds, structure=structure)

    if n_markers == 0 or not foreground.any():
        return np.zeros(foreground.shape, dtype=np.int64)

    labels = watershed(
        image=p_boundary,
        markers=markers,
        mask=foreground,
        connectivity=structure,
        watershed_line=False,
    ).astype(np.int64)

    if min_area > 0:
        counts = np.bincount(labels.ravel())
        too_small = np.flatnonzero(counts < min_area)
        too_small = too_small[too_small != 0]
        if too_small.size > 0:
            labels[np.isin(labels, too_small)] = 0

    return _relabel_consecutive(labels)


def watershed_instance_segmentation_batch(
    prob: ArrayLike,
    interior_threshold: float = 0.50,
    foreground_threshold: float = 0.50,
    min_area: int = 10,
    connectivity: int = 1,
) -> List[np.ndarray]:
    """Aplica a decodificação watershed a um lote ``(B, 3, H, W)``."""
    if isinstance(prob, torch.Tensor):
        prob = prob.detach().cpu().numpy()
    batch = np.asarray(prob)
    if batch.ndim == 3:
        batch = batch[np.newaxis]
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError(f"Esperado shape (B, 3, H, W); recebido {batch.shape}.")

    return [
        watershed_instance_segmentation(
            item,
            interior_threshold=interior_threshold,
            foreground_threshold=foreground_threshold,
            min_area=min_area,
            connectivity=connectivity,
        )
        for item in batch
    ]


def count_instances(labels: np.ndarray) -> int:
    """Conta quantas instâncias existem em uma máscara de rótulos.

    Args:
        labels: Array (H, W) com 0 = fundo e 1..K = instâncias.

    Returns:
        Número de instâncias distintas.
    """
    return int(len(np.unique(labels)) - (1 if 0 in labels else 0))


if __name__ == "__main__":
    # Smoke test: os três comportamentos que definem o método ingênuo.
    probs = np.zeros((40, 40), dtype=np.float32)

    # 1. Dois quadrados separados por uma coluna de fundo.
    probs[:] = 0.0
    probs[10:20, 5:15] = 1.0
    probs[10:20, 20:30] = 1.0
    separados = naive_connected_components(probs, from_logits=False)
    print(f"dois quadrados separados -> {count_instances(separados)} instâncias")
    assert count_instances(separados) == 2

    # 2. Dois quadrados encostados: o método funde os dois. É a falha da Parte 1.
    probs[:] = 0.0
    probs[10:20, 5:15] = 1.0
    probs[10:20, 15:25] = 1.0
    encostados = naive_connected_components(probs, from_logits=False)
    print(f"dois quadrados encostados -> {count_instances(encostados)} instância (falha esperada)")
    assert count_instances(encostados) == 1

    # 3. Contato só na diagonal: separado com conectividade 4, fundido com 8.
    probs[:] = 0.0
    probs[5:10, 5:10] = 1.0
    probs[10:15, 10:15] = 1.0
    conn4 = naive_connected_components(probs, from_logits=False, connectivity=1)
    conn8 = naive_connected_components(probs, from_logits=False, connectivity=2)
    print(f"contato diagonal -> conectividade 4: {count_instances(conn4)} | conectividade 8: {count_instances(conn8)}")
    assert count_instances(conn4) == 2
    assert count_instances(conn8) == 1

    # 4. Filtro de área remove o resíduo e mantém os rótulos consecutivos.
    probs[:] = 0.0
    probs[10:20, 5:15] = 1.0
    probs[30, 30] = 1.0
    com_residuo = naive_connected_components(probs, from_logits=False)
    sem_residuo = naive_connected_components(probs, from_logits=False, min_size=10)
    print(f"com resíduo: {count_instances(com_residuo)} | com min_size=10: {count_instances(sem_residuo)}")
    assert count_instances(com_residuo) == 2
    assert count_instances(sem_residuo) == 1
    assert sorted(np.unique(sem_residuo).tolist()) == [0, 1], "rótulos devem ficar consecutivos"

    # 5. Caminho a partir de logits, como sai da rede.
    logits = torch.full((1, 1, 40, 40), -10.0)
    logits[..., 10:20, 5:15] = 10.0
    de_logits = naive_connected_components(logits)
    print(f"a partir de logits (1, 1, H, W) -> {count_instances(de_logits)} instância")
    assert count_instances(de_logits) == 1

    print("todos os testes ok")
