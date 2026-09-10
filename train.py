"""Pipeline de treino configurável via argumentos CLI.

Treina a baseline de segmentação binária das Partes 0 e 1 e registra em disco
tudo que as partes seguintes vão precisar reproduzir.

Cada execução escreve em `runs/<nome>/`:
- `args.json`: os argumentos exatos daquela execução.
- `history.json`: uma linha por época, com perdas e métricas de validação.
- `per_image.json`: uma linha por imagem de validação da melhor época, com
  contagens e métricas individuais. É a fonte do gráfico de mAP vs densidade
  da Parte 1 e da tabela de média ± desvio da Parte 3.
- `best_model.pth`: pesos da melhor época segundo a métrica de seleção.

Glossário:
- época: uma passada completa por todo o conjunto de treino.
- iteração: uma atualização de pesos, ou seja, um lote processado.
- otimizador: a regra que decide como mexer nos pesos a partir do gradiente.
- learning rate: o tamanho do passo dado a cada atualização.
- seed: número que fixa todo o sorteio, para a execução ser repetível.
- checkpoint: arquivo com os pesos salvos, para usar depois sem retreinar.
"""

from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DSB2018Dataset, SyntheticDataset, create_stratified_splits, load_splits
from src.losses import build_loss, compute_pos_weight, compute_class_weights
from src.metrics import compute_semantic_dice, compute_semantic_iou, evaluate_instances
from src.models import build_model
from src.postprocess import naive_connected_components, watershed_instance_segmentation


def parse_args() -> argparse.Namespace:
    """Define e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treino da baseline de segmentação binária (Partes 0 e 1 do PA1)."
    )

    # Contrato acordado com a dupla
    parser.add_argument("--dataset", choices=["synthetic", "dsb2018"], default="synthetic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # Arquitetura (Eixo 1 das ablações)
    parser.add_argument("--encoder", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--out_channels", type=int, choices=[1, 3], default=1, help="Canais de saída (1=binário, 3=Trilha A watershed).")
    parser.add_argument("--up_mode", choices=["transpose", "bilinear", "nearest"], default="transpose")
    parser.add_argument("--no_skips", action="store_true", help="Desliga as skip connections.")
    parser.add_argument("--no_pretrained", action="store_true", help="Treina o encoder do zero.")

    # Perda (Eixo 2 das ablações)
    parser.add_argument(
        "--loss",
        choices=["bce_dice", "bce", "dice", "focal", "weighted_ce_3c", "multiclass_focal"],
        default="bce_dice",
    )
    parser.add_argument("--gamma", type=float, default=2.0, help="Expoente de foco da focal loss.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Peso da classe positiva na focal.")
    parser.add_argument(
        "--pos_weight",
        default="none",
        help="Peso da classe positiva na BCE: 'none', 'auto' ou um número.",
    )
    parser.add_argument(
        "--class_weights",
        default="auto",
        help="Pesos de classes para perdas multiclasse: 'auto', 'none' ou lista (ex: '1.0,2.5,8.0').",
    )

    # Decodificação em instâncias
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--interior_threshold", type=float, default=0.5, help="Limiar de semente de interior no watershed.")
    parser.add_argument("--foreground_threshold", type=float, default=0.5, help="Limiar de foreground no watershed.")
    parser.add_argument("--min_area", type=int, default=10, help="Área mínima para descartar ruído no watershed.")
    parser.add_argument("--connectivity", type=int, choices=[1, 2], default=1)
    parser.add_argument("--min_size", type=int, default=0)
    parser.add_argument("--matching", choices=["hungarian", "greedy"], default="hungarian")

    # Dados e execução
    parser.add_argument("--num_samples", type=int, default=500, help="Amostras sintéticas por época.")
    parser.add_argument("--target_size", type=int, default=256, help="Lado das imagens do DSB2018.")
    parser.add_argument("--data_dir", default="data/raw/stage1_train")
    parser.add_argument("--splits_path", default="data/splits.json")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="Pasta da execução. Default: runs/<config>.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument(
        "--select_by",
        choices=["mAP", "iou"],
        default="mAP",
        help="Métrica de validação usada para escolher a melhor época.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Fixa todos os geradores aleatórios para tornar a execução repetível.

    Args:
        seed: Semente aplicada a random, NumPy e PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Semeia cada worker do DataLoader, que roda em processo próprio."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Monta os DataLoaders de treino e validação conforme o dataset escolhido.

    Args:
        args: Argumentos da execução.

    Returns:
        Tupla (loader de treino, loader de validação).

    Raises:
        FileNotFoundError: Se o DSB2018 for pedido e os dados não estiverem em disco.
    """
    is_3c = (args.out_channels == 3)
    if args.dataset == "synthetic":
        train_set: Any = SyntheticDataset(num_samples=args.num_samples, seed=None, three_class=is_3c)
        # Validação com seed fixa: as mesmas imagens em toda época e em toda execução.
        val_set: Any = SyntheticDataset(num_samples=100, seed=12345, three_class=is_3c)
    else:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(
                f"Dados do DSB2018 não encontrados em '{args.data_dir}'. "
                "Veja as instruções de download na seção 3 do README."
            )

        if os.path.exists(args.splits_path):
            splits = load_splits(args.splits_path)
        else:
            print(f"Splits ausentes. Criando partição estratificada em '{args.splits_path}'...")
            splits = create_stratified_splits(
                data_dir=args.data_dir, splits_path=args.splits_path, seed=args.seed
            )

        size = (args.target_size, args.target_size)
        train_set = DSB2018Dataset(args.data_dir, image_ids=splits["train"], target_size=size, three_class=is_3c)
        val_set = DSB2018Dataset(args.data_dir, image_ids=splits["val"], target_size=size, three_class=is_3c)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def resolve_pos_weight(spec: str, loader: DataLoader, max_batches: int = 20) -> Optional[float]:
    """Interpreta o argumento --pos_weight, estimando da amostra quando for 'auto'.

    A estimativa usa apenas o loader de treino, nunca validação ou teste.

    Args:
        spec: 'none', 'auto' ou um número em texto.
        loader: DataLoader de treino.
        max_batches: Quantos lotes inspecionar na estimativa.

    Returns:
        Peso da classe positiva, ou None quando desligado.
    """
    if spec == "none":
        return None

    if spec != "auto":
        return float(spec)

    masks: List[torch.Tensor] = []
    for i, batch in enumerate(loader):
        masks.append(batch["mask_semantic"])
        if i + 1 >= max_batches:
            break

    value = compute_pos_weight(torch.cat(masks))
    print(f"pos_weight estimado do treino: {value:.3f}")
    return value


def resolve_class_weights(spec: str, loader: DataLoader, max_batches: int = 20) -> Optional[List[float]]:
    """Interpreta o argumento --class_weights, estimando da amostra quando for 'auto'."""
    if spec == "none":
        return None

    if spec != "auto":
        return [float(x.strip()) for x in spec.split(",")]

    targets: List[torch.Tensor] = []
    for i, batch in enumerate(loader):
        if "mask_three_class" in batch:
            targets.append(batch["mask_three_class"])
        if i + 1 >= max_batches:
            break

    if not targets:
        return [1.0, 2.5, 8.0]

    w = compute_class_weights(torch.cat(targets), num_classes=3)
    # Suavização suave por raiz quadrada para evitar gradientes desestabilizadores
    w_smoothed = (w / w[0]).sqrt()
    weights_list = [round(float(v), 3) for v in w_smoothed]
    print(f"pesos de 3 classes estimados do treino: {weights_list}")
    return weights_list


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str,
    out_channels: int = 1,
) -> float:
    """Roda uma época de treino.

    Args:
        model: Modelo a treinar.
        loader: DataLoader de treino.
        loss_fn: Função de perda.
        optimizer: Otimizador.
        device: Dispositivo de execução.
        desc: Rótulo da barra de progresso.
        out_channels: 1 para segmentação binária, 3 para Trilha A (fundo, interior, fronteira).

    Returns:
        Perda média da época.
    """
    model.train()
    total, n_batches = 0.0, 0

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if out_channels == 3:
            targets = batch["mask_three_class"].to(device, non_blocking=True)
        else:
            targets = batch["mask_semantic"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()

        total += loss.item()
        n_batches += 1

    return total / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Avalia o modelo, coletando métricas semânticas e de instâncias por imagem.

    Args:
        model: Modelo a avaliar.
        loader: DataLoader de validação.
        loss_fn: Função de perda, para acompanhar a perda de validação.
        device: Dispositivo de execução.
        args: Argumentos da execução (limiar, conectividade, matching, out_channels).

    Returns:
        Tupla (métricas agregadas, registros por imagem).
    """
    model.eval()

    total_loss, n_batches = 0.0, 0
    records: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="validação", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if args.out_channels == 3:
            targets = batch["mask_three_class"].to(device, non_blocking=True)
        else:
            targets = batch["mask_semantic"].to(device, non_blocking=True)

        logits = model(images)
        total_loss += loss_fn(logits, targets).item()
        n_batches += 1

        if args.out_channels == 3:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            probs = torch.sigmoid(logits).cpu().numpy()

        gt_semantic = batch["mask_semantic"].numpy()
        gt_instance = batch["mask_instance"].numpy()

        # SyntheticDataset não devolve image_id; DSB2018Dataset devolve.
        ids = batch.get("image_id")

        for i in range(images.shape[0]):
            if args.out_channels == 3:
                pred_labels = watershed_instance_segmentation(
                    probs[i],
                    interior_threshold=args.interior_threshold,
                    foreground_threshold=args.foreground_threshold,
                    min_area=args.min_area,
                    connectivity=args.connectivity,
                )
                pred_binary = ((probs[i, 1] + probs[i, 2]) >= args.foreground_threshold).astype(np.float32)
            else:
                pred_labels = naive_connected_components(
                    probs[i],
                    threshold=args.threshold,
                    from_logits=False,
                    connectivity=args.connectivity,
                    min_size=args.min_size,
                )
                pred_binary = (probs[i] >= args.threshold).astype(np.float32)

            instance_result = evaluate_instances(
                pred_labels, gt_instance[i], method=args.matching
            )

            records.append(
                {
                    "image_id": ids[i] if ids is not None else f"{len(records):05d}",
                    "iou": compute_semantic_iou(pred_binary, gt_semantic[i]),
                    "dice": compute_semantic_dice(pred_binary, gt_semantic[i]),
                    "mAP": instance_result["mAP"],
                    "count_error": instance_result["count_error"],
                    "n_pred": instance_result["n_pred"],
                    "n_gt": instance_result["n_gt"],
                }
            )

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "iou": float(np.mean([r["iou"] for r in records])),
        "dice": float(np.mean([r["dice"] for r in records])),
        "mAP": float(np.mean([r["mAP"] for r in records])),
        "count_error": float(np.mean([r["count_error"] for r in records])),
    }

    return metrics, records


def save_checkpoint(path: str, model: torch.nn.Module, args: argparse.Namespace,
                    epoch: int, metrics: Dict[str, float]) -> None:
    """Salva os pesos no formato esperado por `evaluate.py` e pelo notebook de inferência.

    Só tipos primitivos entram no arquivo: a partir do PyTorch 2.6 o `torch.load`
    usa `weights_only=True` por padrão e recusa objetos como o Namespace do argparse.

    Args:
        path: Caminho do arquivo .pth.
        model: Modelo cujos pesos serão salvos.
        args: Argumentos da execução.
        epoch: Época correspondente aos pesos.
        metrics: Métricas de validação daquela época.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_args": {
                "encoder": args.encoder,
                "out_channels": args.out_channels,
                "up_mode": args.up_mode,
                "use_skips": not args.no_skips,
                "pretrained": not args.no_pretrained,
            },
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = args.out or os.path.join(
        "runs", f"{args.dataset}_{args.loss}_{args.up_mode}_seed{args.seed}"
    )
    os.makedirs(run_name, exist_ok=True)

    device = torch.device(args.device)
    print(f"dataset={args.dataset} | perda={args.loss} | device={device} | saída={run_name}")

    train_loader, val_loader = build_dataloaders(args)
    print(f"treino: {len(train_loader.dataset)} imagens | validação: {len(val_loader.dataset)} imagens")

    model = build_model(
        encoder=args.encoder,
        out_channels=args.out_channels,
        up_mode=args.up_mode,
        use_skips=not args.no_skips,
        pretrained=not args.no_pretrained,
    ).to(device)

    loss_kwargs: Dict[str, Any] = {}
    if args.loss == "focal":
        loss_kwargs = {"alpha": args.alpha, "gamma": args.gamma}
    elif args.loss in ("weighted_ce_3c", "multiclass_focal"):
        weights = resolve_class_weights(args.class_weights, train_loader)
        if args.loss == "weighted_ce_3c":
            loss_kwargs = {"class_weights": weights}
        else:
            loss_kwargs = {"alpha": weights, "gamma": args.gamma}
    else:
        loss_kwargs = {"pos_weight": resolve_pos_weight(args.pos_weight, train_loader)}
    loss_fn = build_loss(args.loss, **loss_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with open(os.path.join(run_name, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    history: List[Dict[str, Any]] = []
    best_score = -float("inf")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            f"época {epoch}/{args.epochs}",
            out_channels=args.out_channels,
        )
        val_metrics, val_records = validate(model, val_loader, loss_fn, device, args)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_mAP": val_metrics["mAP"],
            "val_count_error": val_metrics["count_error"],
            "seconds": time.time() - epoch_started,
        }
        history.append(entry)

        print(
            f"época {epoch:3d} | treino {train_loss:.4f} | val {val_metrics['loss']:.4f} "
            f"| IoU {val_metrics['iou']:.4f} | Dice {val_metrics['dice']:.4f} "
            f"| mAP {val_metrics['mAP']:.4f} | erro contagem {val_metrics['count_error']:.2f} "
            f"| {entry['seconds']:.1f}s"
        )

        score = val_metrics[args.select_by]
        if score > best_score:
            best_score = score
            save_checkpoint(os.path.join(run_name, "best_model.pth"), model, args, epoch, val_metrics)
            save_checkpoint(args.checkpoint, model, args, epoch, val_metrics)
            with open(os.path.join(run_name, "per_image.json"), "w", encoding="utf-8") as f:
                json.dump(val_records, f, indent=2, ensure_ascii=False)

        with open(os.path.join(run_name, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - started
    print(f"\nconcluído em {elapsed / 60:.2f} min | melhor {args.select_by}: {best_score:.4f}")
    print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
