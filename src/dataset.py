"""Módulo de datasets, gerador sintético e particionamento estratificado.

Este módulo implementa:
1. SyntheticDataset: Gerador procedural de dados sintéticos para a Parte 0 (elipses
   com sobreposição, ruído e contraste variáveis em canvas 128x128).
2. DSB2018Dataset: Leitor e pré-processador da base real Data Science Bowl 2018
   (BBBC038v1) com fusão das máscaras individuais de instâncias.
3. Funções de particionamento estratificado (create_stratified_splits e load_splits)
   baseadas na modalidade de microscopia informada em metadata.xlsx.

Todos os datasets respeitam estritamente os contratos de interface:
- image: Tensor float32 (3, H, W) normalizado em [0.0, 1.0].
- mask_semantic: Tensor float32 (1, H, W) binário {0.0, 1.0}.
- mask_instance: Tensor int64 (H, W) onde 0 = fundo e 1..K = IDs de instâncias.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import glob
import json
import os
from PIL import Image
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, generate_binary_structure
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# Dataset Sintético (Parte 0: Teste Unitário Sintético)

def generate_synthetic_sample(
    h: int = 128,
    w: int = 128,
    min_ellipses: int = 5,
    max_ellipses: int = 20,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera proceduralmente uma imagem sintética com elipses se tocando e suas máscaras.

    Atende aos requisitos da Parte 0 do PA1:
    - Imagens 128 x 128.
    - 5 a 20 elipses de tamanhos variados.
    - Muitas elipses se tocando / sobrepondo.
    - Ruído e contraste variáveis.

    Args:
        h: Altura da imagem (padrão 128).
        w: Largura da imagem (padrão 128).
        min_ellipses: Quantidade mínima de elipses por imagem (padrão 5).
        max_ellipses: Quantidade máxima de elipses por imagem (padrão 20).
        rng: Gerador de números aleatórios do NumPy (opcional para reprodutibilidade).

    Returns:
        Tupla (image, mask_semantic, mask_instance):
            - image: Array numpy float32 (3, h, w) em [0.0, 1.0].
            - mask_semantic: Array numpy float32 (1, h, w) binário {0.0, 1.0}.
            - mask_instance: Array numpy int64 (h, w) com 0 = fundo e 1..K = IDs.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_ellipses = rng.randint(min_ellipses, max_ellipses + 1)
    canvas = np.zeros((h, w), dtype=np.float32)
    instance_mask = np.zeros((h, w), dtype=np.int64)

    # Nível base do fundo (fundo escuro de microscopia de fluorescência)
    bg_level = rng.uniform(0.05, 0.20)
    canvas[:] = bg_level

    # Grades de coordenadas para vetorização
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    centers: List[Tuple[float, float, float]] = []
    current_inst_id = 1

    for _ in range(n_ellipses):
        # Semi-eixos variados
        a = rng.uniform(5.0, 18.0)
        b = rng.uniform(4.0, 16.0)
        theta = rng.uniform(0.0, np.pi)

        # 60% de chance de nascer próxima a um núcleo existente para incentivar contato/toque
        max_r = max(a, b)
        if len(centers) > 0 and rng.rand() < 0.60:
            ref_cy, ref_cx, ref_r = centers[rng.choice(len(centers))]
            touch_dist = (ref_r + max_r) * rng.uniform(0.70, 1.15)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            cy = float(np.clip(ref_cy + touch_dist * np.sin(angle), max_r, h - max_r))
            cx = float(np.clip(ref_cx + touch_dist * np.cos(angle), max_r, w - max_r))
        else:
            cy = rng.uniform(max_r, h - max_r)
            cx = rng.uniform(max_r, w - max_r)

        centers.append((cy, cx, max_r))

        # Equação analítica da elipse rotacionada
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = (x_coords - cx) * cos_t + (y_coords - cy) * sin_t
        y_rot = -(x_coords - cx) * sin_t + (y_coords - cy) * cos_t
        dist_sq = (x_rot / a) ** 2 + (y_rot / b) ** 2
        ellipse_mask = dist_sq <= 1.0

        if not np.any(ellipse_mask):
            continue

        # Perfil de intensidade do núcleo (mais brilhante no centro, com variação por núcleo)
        peak_brightness = rng.uniform(0.65, 0.98)
        profile = peak_brightness * (1.0 - 0.35 * dist_sq[ellipse_mask])
        canvas[ellipse_mask] = np.maximum(canvas[ellipse_mask], profile)

        # Máscara de instâncias com ID único
        instance_mask[ellipse_mask] = current_inst_id
        current_inst_id += 1

    # Adição de ruído gaussiano
    noise_sigma = rng.uniform(0.02, 0.05)
    noise = rng.normal(0.0, noise_sigma, (h, w)).astype(np.float32)
    canvas = np.clip(canvas + noise, 0.0, 1.0)

    # Variação aleatória de contraste e brilho global
    contrast_scale = rng.uniform(0.85, 1.25)
    brightness_shift = rng.uniform(-0.05, 0.05)
    canvas = np.clip((canvas - 0.5) * contrast_scale + 0.5 + brightness_shift, 0.0, 1.0)

    # Máscara semântica binária (1 no foreground, 0 no fundo)
    semantic_mask = (instance_mask > 0).astype(np.float32)[np.newaxis, :, :]  # (1, H, W)

    # Replicar para 3 canais (3, H, W) para compatibilidade com encoders pré-treinados (ex: ResNet)
    image_3ch = np.repeat(canvas[np.newaxis, :, :], 3, axis=0)

    return image_3ch, semantic_mask, instance_mask


def generate_three_class_mask(
    instance_mask: Union[np.ndarray, torch.Tensor],
    connectivity: int = 1,
) -> np.ndarray:
    """Gera uma máscara com 3 classes para a Trilha A (Fronteiras e Watershed).

    Classes geradas:
        - 0 (Fundo): Pixels externos a qualquer núcleo.
        - 1 (Interior): Núcleos individuais erodidos morfologicamente por 1 pixel.
                        Garante que núcleos vizinhos tenham sementes desconectadas.
        - 2 (Fronteira): Regiões de contorno e contato entre núcleos adjacentes.

    Args:
        instance_mask: Array numpy ou Tensor int64 (H, W) com 0=fundo e 1..K=IDs.
        connectivity: Conectividade do elemento estruturante (1 = cruz 4-conexo,
                      2 = quadrado 3x3 8-conexo). Padrão: 1.

    Returns:
        Array numpy int64 (H, W) contendo valores em {0, 1, 2}.
    """
    if isinstance(instance_mask, torch.Tensor):
        inst = instance_mask.detach().cpu().numpy()
    else:
        inst = np.asarray(instance_mask)

    three_class = np.zeros_like(inst, dtype=np.int64)
    unique_ids = np.unique(inst)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        return three_class

    struct = generate_binary_structure(2, connectivity)
    h, w = inst.shape

    for uid in unique_ids:
        ys, xs = np.where(inst == uid)
        y0, y1 = max(0, ys.min() - 1), min(h, ys.max() + 2)
        x0, x1 = max(0, xs.min() - 1), min(w, xs.max() + 2)

        sub_m = (inst[y0:y1, x0:x1] == uid)
        sub_eroded = binary_erosion(sub_m, structure=struct)

        # Se o núcleo for muito pequeno e a erosão o apagar, preserva o pixel central
        if not np.any(sub_eroded):
            sub_coords = np.argwhere(sub_m)
            center = sub_coords[len(sub_coords) // 2]
            sub_eroded[center[0], center[1]] = True

        three_class[y0:y1, x0:x1][sub_eroded] = 1

    # Classe 2 (Fronteira): pixels pertencentes a núcleos que não são interior
    three_class[(inst > 0) & (three_class != 1)] = 2

    return three_class


class SyntheticDataset(Dataset):
    """PyTorch Dataset para geração procedural de amostras sintéticas na Parte 0.

    Pode gerar dados dinamicamente a cada época ou manter um conjunto determinístico
    via seed (recomendado para validação).
    """

    def __init__(
        self,
        num_samples: int = 500,
        height: int = 128,
        width: int = 128,
        min_ellipses: int = 5,
        max_ellipses: int = 20,
        seed: Optional[int] = None,
        three_class: bool = False,
    ) -> None:
        """Inicializa o SyntheticDataset.

        Args:
            num_samples: Quantidade de amostras por época.
            height: Altura das imagens (128).
            width: Largura das imagens (128).
            min_ellipses: Número mínimo de elipses (5).
            max_ellipses: Número máximo de elipses (20).
            seed: Semente opcional para dados reproduzíveis (ex: validação fixa).
            three_class: Se True, adiciona 'mask_three_class' com fundo/interior/fronteira.
        """
        super().__init__()
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.min_ellipses = min_ellipses
        self.max_ellipses = max_ellipses
        self.seed = seed
        self.three_class = three_class

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retorna uma amostra contendo imagem e máscaras.

        Returns:
            Dict com:
                - 'image': Tensor float32 (3, H, W) em [0.0, 1.0].
                - 'mask_semantic': Tensor float32 (1, H, W) binário {0.0, 1.0}.
                - 'mask_instance': Tensor int64 (H, W) com IDs de instâncias.
                - 'mask_three_class' (opcional): Tensor int64 (H, W) com classes {0, 1, 2}.
        """
        # Se seed estiver definido, gera de forma determinística por índice
        rng = np.random.RandomState(self.seed + idx) if self.seed is not None else None

        img, sem, inst = generate_synthetic_sample(
            h=self.height,
            w=self.width,
            min_ellipses=self.min_ellipses,
            max_ellipses=self.max_ellipses,
            rng=rng,
        )

        sample: Dict[str, torch.Tensor] = {
            "image": torch.from_numpy(img).float(),
            "mask_semantic": torch.from_numpy(sem).float(),
            "mask_instance": torch.from_numpy(inst).long(),
        }

        if self.three_class:
            mask_3c = generate_three_class_mask(inst)
            sample["mask_three_class"] = torch.from_numpy(mask_3c).long()

        return sample


# Dataset Real (Parte 1: DSB2018 / BBBC038v1)

class DSB2018Dataset(Dataset):
    """PyTorch Dataset para os dados reais do Data Science Bowl 2018.

    Carrega as imagens originais de microscopia e funde as dezenas/centenas
    de máscaras PNG individuais de cada núcleo em:
    - Uma máscara de instâncias int64 (H, W)
    - Uma máscara semântica binária float32 (1, H, W)
    """

    def __init__(
        self,
        root_dir: str = "data/raw/stage1_train",
        image_ids: Optional[Sequence[str]] = None,
        target_size: Optional[Tuple[int, int]] = (256, 256),
        transform: Optional[Callable] = None,
        three_class: bool = False,
    ) -> None:
        """Inicializa o DSB2018Dataset.

        Args:
            root_dir: Diretório raiz contendo as pastas das imagens de stage1_train.
            image_ids: Lista opcional de IDs de imagens a incluir (para splits).
                       Se None, carrega todos os diretórios encontrados em root_dir.
            target_size: Tupla opcional (H, W) para redimensionar (padrão 256x256).
            transform: Função ou composição de data augmentation opcional.
            three_class: Se True, adiciona 'mask_three_class' com {0: fundo, 1: interior, 2: fronteira}.
        """
        super().__init__()
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        self.three_class = three_class

        if image_ids is not None:
            self.image_ids = list(image_ids)
        else:
            if os.path.exists(root_dir):
                self.image_ids = sorted([
                    d for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))
                ])
            else:
                self.image_ids = []

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_id = self.image_ids[idx]
        image_dir = os.path.join(self.root_dir, image_id)

        # 1. Carregar imagem original e converter para RGB
        img_file = os.path.join(image_dir, "images", f"{image_id}.png")
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Arquivo de imagem não encontrado: {img_file}")

        with Image.open(img_file) as pil_img:
            img_rgb = pil_img.convert("RGB")
            orig_w, orig_h = img_rgb.size

            if self.target_size is not None:
                img_rgb = img_rgb.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)

            img_np = np.array(img_rgb, dtype=np.float32) / 255.0  # (H, W, 3)

        # 2. Carregar e fundir todas as máscaras individuais de núcleos
        masks_dir = os.path.join(image_dir, "masks")
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

        curr_h = self.target_size[0] if self.target_size is not None else orig_h
        curr_w = self.target_size[1] if self.target_size is not None else orig_w
        instance_mask = np.zeros((curr_h, curr_w), dtype=np.int64)

        for inst_id, m_file in enumerate(mask_files, start=1):
            with Image.open(m_file) as pil_mask:
                if self.target_size is not None:
                    # Redimensionamento de máscaras categóricas deve ser estritamente NEAREST
                    pil_mask = pil_mask.resize(
                        (self.target_size[1], self.target_size[0]),
                        Image.Resampling.NEAREST
                    )
                m_np = np.array(pil_mask) > 0
                instance_mask[m_np] = inst_id

        # 3. Gerar máscara semântica binária (1, H, W)
        semantic_mask = (instance_mask > 0).astype(np.float32)[np.newaxis, :, :]

        # 4. Transpor imagem para formato PyTorch (3, H, W)
        image_chw = np.transpose(img_np, (2, 0, 1))

        # 5. Aplicação de transformações/augmentations adicionais se houver
        if self.transform is not None:
            # Suporte para Albumentations ou funções customizadas
            transformed = self.transform(image=image_chw, mask=semantic_mask, instance=instance_mask)
            image_chw = transformed.get("image", image_chw)
            semantic_mask = transformed.get("mask", semantic_mask)
            instance_mask = transformed.get("instance", instance_mask)

        sample_dict: Dict[str, Union[torch.Tensor, str]] = {
            "image": torch.from_numpy(image_chw).float(),
            "mask_semantic": torch.from_numpy(semantic_mask).float(),
            "mask_instance": torch.from_numpy(instance_mask).long(),
            "image_id": image_id,
        }

        if self.three_class:
            mask_3c = generate_three_class_mask(instance_mask)
            sample_dict["mask_three_class"] = torch.from_numpy(mask_3c).long()

        return sample_dict


def compute_three_class_weights(
    dataset: Dataset,
    num_samples: int = 100,
    smooth_power: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula frequências de classes e pesos normalizados para a Trilha A.

    Args:
        dataset: Dataset que retorna 'mask_instance' ou 'mask_three_class'.
        num_samples: Quantidade máxima de amostras a inspecionar para estimativa.
        smooth_power: Expoente para suavização (1.0 = inverso puro, 0.5 = raiz quadrada).

    Returns:
        Tupla (class_frequencies, normalized_weights):
            - class_frequencies: Array float64 (3,) com a proporção de cada classe.
            - normalized_weights: Array float32 (3,) com pesos normalizados (peso da classe 0 = 1.0).
    """
    total_counts = np.zeros(3, dtype=np.int64)
    n = min(len(dataset), num_samples)

    for i in range(n):
        sample = dataset[i]
        if "mask_three_class" in sample:
            m = sample["mask_three_class"]
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
        else:
            inst = sample["mask_instance"]
            m = generate_three_class_mask(inst)

        for c in range(3):
            total_counts[c] += (m == c).sum()

    total_pixels = total_counts.sum()
    if total_pixels == 0:
        return np.ones(3) / 3.0, np.ones(3, dtype=np.float32)

    freqs = total_counts / float(total_pixels)
    inv_freq = (1.0 / (freqs + 1e-6)) ** smooth_power
    weights = inv_freq / inv_freq[0]

    return freqs, weights.astype(np.float32)


# Estratificação e Criação de Splits (Parte 1)

def identify_image_modality(
    image_path: str,
) -> str:
    """Classifica a modalidade visual da imagem de microscopia do DSB2018.

    Categorias:
    - 'TissueBW': Tecido cervical de grande resolução (1024x1024).
    - 'Color_Histology': Imagens coradas histologicamente (Púrpura / Rosa).
    - 'Default': Microscopia de fluorescência padrão em tons de cinza / fundo escuro.

    Args:
        image_path: Caminho para o arquivo PNG da imagem.

    Returns:
        String representando a modalidade para estratificação.
    """
    with Image.open(image_path) as img:
        w, h = img.size
        arr = np.array(img.convert("RGB"), dtype=np.float32)

    # Resolução 1024x1024 corresponde ao conjunto ISBI TissueBW
    if (w, h) == (1024, 1024):
        return "TissueBW"

    # Avaliação de saturação/diferença de cor entre canais
    diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1]).mean()
    diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2]).mean()

    if diff_rg > 5.0 or diff_rb > 5.0:
        return "Color_Histology"

    return "Default"


def create_stratified_splits(
    data_dir: str = "data/raw/stage1_train",
    metadata_path: str = "data/raw/metadata.xlsx",
    splits_path: str = "data/splits.json",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Gera e salva em disco o particionamento estratificado do dataset DSB2018.

    A estratificação atende à exigência do PA1:
    'split treino/validação/teste estratificado por modalidade ou cidade, justificado na apresentação.'

    Args:
        data_dir: Diretório raiz de stage1_train.
        metadata_path: Caminho para metadata.xlsx (opcional).
        splits_path: Caminho onde o arquivo JSON com os splits será salvo.
        train_ratio: Proporção de treino (padrão 0.80).
        val_ratio: Proporção de validação (padrão 0.10).
        test_ratio: Proporção de teste (padrão 0.10).
        seed: Semente pseudoaleatória para reprodutibilidade estrita.

    Returns:
        Dicionário com listas de image_ids: {'train': [...], 'val': [...], 'test': [...]}
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")

    image_ids = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if len(image_ids) == 0:
        raise ValueError(f"Nenhuma imagem encontrada em {data_dir}.")

    # Mapear cada imagem para sua modalidade
    labels: List[str] = []
    for i_id in image_ids:
        img_file = os.path.join(data_dir, i_id, "images", f"{i_id}.png")
        modality = identify_image_modality(img_file)
        labels.append(modality)

    # 1. Separar Treino vs (Validação + Teste) estratificado
    temp_ratio = val_ratio + test_ratio
    train_ids, temp_ids, _, temp_labels = train_test_split(
        image_ids,
        labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=seed,
    )

    # 2. Separar Validação vs Teste estratificado
    val_rel_ratio = val_ratio / temp_ratio
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1.0 - val_rel_ratio),
        stratify=temp_labels,
        random_state=seed,
    )

    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    # Salvar em arquivo JSON
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return splits


def load_splits(splits_path: str = "data/splits.json") -> Dict[str, List[str]]:
    """Carrega os splits de particionamento do arquivo JSON.

    Args:
        splits_path: Caminho do arquivo de splits.

    Returns:
        Dicionário com as listas de IDs para 'train', 'val' e 'test'.
    """
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Arquivo de splits não encontrado: {splits_path}. Execute create_stratified_splits() primeiro.")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    return splits