import argparse
import json
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50
from tqdm.auto import tqdm


@dataclass
class Config:
    data_list: str = "./mchar_data_list_0515.csv"
    dataset_dir: str = "./dataset"
    checkpoints: str = "./checkpoints"
    result_csv: str = "./result.csv"
    arch: str = "resnet50"
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    class_num: int = 11
    max_length: int = 4
    image_height: int = 128
    image_width: int = 224
    eval_interval: int = 1
    print_interval: int = 50
    smooth: float = 0.1
    erase_prob: float = 0.25
    pretrained: bool = True
    seed: int = 2026
    num_workers: int = 0 if os.name == "nt" else 4
    char_arch: str = "resnet18"
    char_class_num: int = 10
    char_batch_size: int = 256
    char_epochs: int = 25
    char_lr: float = 5e-4
    char_weight_decay: float = 1e-4
    char_image_size: int = 64
    bbox_pad_ratio: float = 0.15
    yolo_output_dir: str = "./dataset/yolo_digit"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5


config = Config()


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def data_paths(dataset_dir: str | Path = config.dataset_dir) -> dict[str, Path]:
    root = Path(dataset_dir)
    return {
        "train_data": root / "mchar_train",
        "val_data": root / "mchar_val",
        "test_data": root / "mchar_test_a",
        "train_label": root / "mchar_train.json",
        "val_label": root / "mchar_val.json",
        "submit_file": root / "mchar_sample_submit_A.csv",
    }


def download_file(url: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and save_path.stat().st_size > 0:
        print(f"Skip existing file: {save_path}")
        return

    print(f"Downloading {save_path.name} from {url}")
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with save_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def prepare_dataset(
    data_list: str | Path = config.data_list,
    dataset_dir: str | Path = config.dataset_dir,
) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    links = pd.read_csv(data_list)
    for _, row in links.iterrows():
        download_file(row["link"], dataset_dir / row["file"])

    for zip_name in ["mchar_train", "mchar_val", "mchar_test_a"]:
        target_dir = dataset_dir / zip_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"Skip extracted folder: {target_dir}")
            continue
        print(f"Extracting {zip_name}.zip")
        with zipfile.ZipFile(dataset_dir / f"{zip_name}.zip", "r") as zip_file:
            zip_file.extractall(dataset_dir)

    macos_dir = dataset_dir / "__MACOSX"
    if macos_dir.exists():
        shutil.rmtree(macos_dir)

    paths = data_paths(dataset_dir)
    print(f"train image counts: {len(list(paths['train_data'].glob('*.png')))}")
    print(f"val image counts: {len(list(paths['val_data'].glob('*.png')))}")
    print(f"test image counts: {len(list(paths['test_data'].glob('*.png')))}")
    return paths


def load_labels(label_path: str | Path) -> dict:
    with Path(label_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def show_train_example(dataset_dir: str | Path = config.dataset_dir, image_name: str = "000000.png") -> None:
    paths = data_paths(dataset_dir)
    labels = load_labels(paths["train_label"])
    print(labels[image_name])


def show_submit_example(dataset_dir: str | Path = config.dataset_dir) -> None:
    paths = data_paths(dataset_dir)
    print(pd.read_csv(paths["submit_file"]).head())


def summarize_label_lengths(dataset_dir: str | Path = config.dataset_dir) -> dict[int, int]:
    labels = load_labels(data_paths(dataset_dir)["train_label"])
    summary: dict[int, int] = {}
    for mark in labels.values():
        length = len(mark["label"])
        summary[length] = summary.get(length, 0) + 1

    for length, count in sorted(summary.items()):
        print(f"{length}个数字的图片数目: {count}")
    return summary


def plot_image_sizes(dataset_dir: str | Path = config.dataset_dir) -> None:
    sizes = [Image.open(img).size for img in data_paths(dataset_dir)["train_data"].glob("*.png")]
    sizes = np.array(sizes)
    plt.figure(figsize=(10, 8))
    plt.scatter(sizes[:, 0], sizes[:, 1], s=8)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("image width-height summary")
    plt.show()


def plot_bbox_sizes(dataset_dir: str | Path = config.dataset_dir) -> None:
    labels = load_labels(data_paths(dataset_dir)["train_label"])
    bboxes = []
    for mark in labels.values():
        for i in range(len(mark["label"])):
            bboxes.append([mark["left"][i], mark["top"][i], mark["width"][i], mark["height"][i]])

    bboxes = np.array(bboxes)
    _, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(bboxes[:, 2], bboxes[:, 3], s=8)
    ax.set_title("bbox width-height summary")
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    plt.show()


def run_eda(dataset_dir: str | Path = config.dataset_dir) -> None:
    show_train_example(dataset_dir)
    show_submit_example(dataset_dir)
    summarize_label_lengths(dataset_dir)
    plot_image_sizes(dataset_dir)
    plot_bbox_sizes(dataset_dir)


def _make_loader_kwargs(num_workers: int, device: torch.device) -> dict[str, Any]:
    return {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }


def _clamp_box(left: float, top: float, width: float, height: float, img_w: int, img_h: int, pad_ratio: float):
    pad_w = width * pad_ratio
    pad_h = height * pad_ratio
    x1 = int(max(0, np.floor(left - pad_w)))
    y1 = int(max(0, np.floor(top - pad_h)))
    x2 = int(min(img_w, np.ceil(left + width + pad_w)))
    y2 = int(min(img_h, np.ceil(top + height + pad_h)))
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)
    return x1, y1, x2, y2


def crop_digit(image: Image.Image, bbox: tuple[float, float, float, float], pad_ratio: float) -> Image.Image:
    left, top, width, height = bbox
    x1, y1, x2, y2 = _clamp_box(left, top, width, height, image.width, image.height, pad_ratio)
    return image.crop((x1, y1, x2, y2))


class DigitsDataset(Dataset):
    def __init__(
        self,
        mode: str = "train",
        dataset_dir: str | Path = config.dataset_dir,
        image_size: tuple[int, int] = (config.image_height, config.image_width),
        max_length: int = config.max_length,
        aug: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.aug = aug
        self.image_size = image_size
        self.max_length = max_length
        self.paths = data_paths(dataset_dir)

        if mode == "test":
            self.samples = sorted(self.paths["test_data"].glob("*.png"))
            return

        labels = load_labels(self.paths[f"{mode}_label"])
        imgs = sorted(self.paths[f"{mode}_data"].glob("*.png"))
        self.samples = [(img, labels[img.name]) for img in imgs if img.name in labels]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.mode == "test":
            img_path = self.samples[idx]
            label = None
        else:
            img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform()(image)

        if self.mode == "test":
            return image, str(img_path)

        digits = label["label"][: self.max_length]
        digits = digits + [10] * (self.max_length - len(digits))
        return image, torch.tensor(digits, dtype=torch.long)

    def transform(self):
        ops = [
            transforms.Resize((self.image_size[0], 256)),
            transforms.CenterCrop(self.image_size),
        ]
        if self.aug:
            ops.extend(
                [
                    transforms.ColorJitter(0.15, 0.15, 0.15),
                    transforms.RandomGrayscale(0.1),
                    transforms.RandomAffine(10, translate=(0.05, 0.08), shear=4),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if self.aug and config.erase_prob > 0:
            ops.append(transforms.RandomErasing(p=config.erase_prob, scale=(0.02, 0.08)))
        return transforms.Compose(ops)


class CharCropDataset(Dataset):
    """将门牌图片按 bbox 裁成单字符样本，用于字符分类训练。"""

    def __init__(
        self,
        mode: str = "train",
        dataset_dir: str | Path = config.dataset_dir,
        image_size: int = config.char_image_size,
        pad_ratio: float = config.bbox_pad_ratio,
        aug: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.image_size = image_size
        self.pad_ratio = pad_ratio
        self.aug = aug
        self.paths = data_paths(dataset_dir)
        self.samples: list[dict[str, Any]] = []
        labels = load_labels(self.paths[f"{mode}_label"])
        image_paths = sorted(self.paths[f"{mode}_data"].glob("*.png"))

        for image_path in image_paths:
            mark = labels.get(image_path.name)
            if mark is None:
                continue
            digits = mark["label"]
            for idx, digit in enumerate(digits):
                self.samples.append(
                    {
                        "image_path": image_path,
                        "digit": int(digit),
                        "bbox": (
                            float(mark["left"][idx]),
                            float(mark["top"][idx]),
                            float(mark["width"][idx]),
                            float(mark["height"][idx]),
                        ),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def transform(self):
        ops = [transforms.Resize((self.image_size, self.image_size))]
        if self.aug:
            ops.extend(
                [
                    transforms.RandomApply([transforms.RandomRotation(8)], p=0.5),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
                    transforms.ColorJitter(0.2, 0.2, 0.2),
                    transforms.RandomGrayscale(0.1),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(ops)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = crop_digit(image, sample["bbox"], self.pad_ratio)
        image = self.transform()(image)
        return image, torch.tensor(sample["digit"], dtype=torch.long)


def visualize_char_crops(
    dataset_dir: str | Path = config.dataset_dir,
    mode: str = "train",
    sample_count: int = 12,
    pad_ratio: float = config.bbox_pad_ratio,
) -> None:
    dataset = CharCropDataset(mode=mode, dataset_dir=dataset_dir, aug=False, pad_ratio=pad_ratio)
    sample_count = min(sample_count, len(dataset))
    indices = np.random.choice(len(dataset), sample_count, replace=False)
    cols = 4
    rows = int(np.ceil(sample_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, indices):
        img, label = dataset[idx]
        arr = img.permute(1, 2, 0).numpy()
        arr = (arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        ax.imshow(arr)
        ax.set_title(f"digit={int(label)}")
    plt.tight_layout()
    plt.show()


def build_resnet_backbone(arch: str, pretrained: bool) -> tuple[nn.Module, int]:
    model_fns = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
    if arch not in model_fns:
        raise ValueError(f"Unsupported arch: {arch}")
    model_fn = model_fns[arch]
    try:
        model = model_fn(weights="DEFAULT" if pretrained else None)
    except TypeError:
        model = model_fn(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, in_features


class StreetNumberRecognizer(nn.Module):
    def __init__(
        self,
        arch: str = config.arch,
        class_num: int = config.class_num,
        max_length: int = config.max_length,
        pretrained: bool = config.pretrained,
    ):
        super().__init__()
        self.backbone, in_features = build_resnet_backbone(arch, pretrained)
        self.heads = nn.ModuleList([nn.Linear(in_features, class_num) for _ in range(max_length)])

    def forward(self, img):
        feat = self.backbone(img)
        return tuple(head(feat) for head in self.heads)


class CharClassifier(nn.Module):
    def __init__(self, arch: str = config.char_arch, class_num: int = config.char_class_num, pretrained: bool = True):
        super().__init__()
        self.backbone, in_features = build_resnet_backbone(arch, pretrained)
        self.fc = nn.Linear(in_features, class_num)

    def forward(self, img):
        feat = self.backbone(img)
        return self.fc(feat)


class LabelSmoothEntropy(nn.Module):
    def __init__(self, smooth: float = config.smooth, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, preds, targets):
        class_num = preds.shape[1]
        lb_pos = 1.0 - self.smooth
        lb_neg = self.smooth / (class_num - 1)
        smoothed = torch.full_like(preds, lb_neg).scatter_(1, targets[:, None], lb_pos)
        loss = -(F.log_softmax(preds, dim=1) * smoothed).sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class Trainer:
    def __init__(self, cfg: Config = config, val: bool = True):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader_kwargs = _make_loader_kwargs(cfg.num_workers, self.device)

        self.train_set = DigitsDataset(mode="train", dataset_dir=cfg.dataset_dir, aug=True)
        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs
        )
        self.val_loader = None
        if val:
            self.val_loader = DataLoader(
                DigitsDataset(mode="val", dataset_dir=cfg.dataset_dir, aug=False),
                batch_size=cfg.batch_size,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
        self.model = StreetNumberRecognizer(
            arch=cfg.arch, class_num=cfg.class_num, max_length=cfg.max_length, pretrained=cfg.pretrained
        ).to(self.device)
        self.criterion = LabelSmoothEntropy(cfg.smooth).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.best_acc = 0.0
        self.best_checkpoint_path = ""

    def train(self) -> str:
        for epoch in range(self.cfg.epochs):
            self.train_epoch(epoch)
            if self.val_loader is not None and (epoch + 1) % self.cfg.eval_interval == 0:
                print("Start Evaluation")
                acc = self.eval()
                if acc > self.best_acc:
                    self.best_acc = acc
                    Path(self.cfg.checkpoints).mkdir(parents=True, exist_ok=True)
                    save_path = Path(self.cfg.checkpoints) / f"epoch-{self.cfg.arch}-{epoch + 1}-acc-{acc * 100:.2f}.pth"
                    self.save_model(save_path)
                    self.best_checkpoint_path = str(save_path)
                    print(f"{save_path} saved successfully.")
        return self.best_checkpoint_path

    def train_epoch(self, epoch: int) -> float:
        total_loss = 0.0
        corrects = 0
        seen = 0
        self.model.train()
        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.epochs}")
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            pred = self.model(img)
            loss = sum(self.criterion(p, label[:, idx]) for idx, p in enumerate(pred))
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(epoch + i / max(len(self.train_loader), 1))
            seen += img.size(0)
            corrects += self._count_correct(pred, label)
            total_loss += loss.item()
            if (i + 1) % self.cfg.print_interval == 0 or i == 0:
                tbar.set_postfix(loss=total_loss / (i + 1), acc=corrects / seen)
        return corrects / seen

    def eval(self) -> float:
        if self.val_loader is None:
            raise RuntimeError("Validation loader is not initialized.")
        self.model.eval()
        corrects = 0
        seen = 0
        with torch.no_grad():
            tbar = tqdm(self.val_loader, desc="Validation")
            for img, label in tbar:
                img = img.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                pred = self.model(img)
                seen += img.size(0)
                corrects += self._count_correct(pred, label)
                tbar.set_postfix(acc=corrects / seen)
        acc = corrects / seen
        print(f"Val Acc: {acc:.4f}")
        self.model.train()
        return acc

    @staticmethod
    def _count_correct(pred, label) -> int:
        matched = torch.stack([p.argmax(dim=1) == label[:, idx] for idx, p in enumerate(pred)], dim=1)
        return torch.all(matched, dim=1).sum().item()

    def save_model(self, save_path: str | Path) -> None:
        torch.save(
            {"model": self.model.state_dict(), "arch": self.cfg.arch, "class_num": self.cfg.class_num, "max_length": self.cfg.max_length},
            save_path,
        )


class CharTrainer:
    def __init__(self, cfg: Config = config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader_kwargs = _make_loader_kwargs(cfg.num_workers, self.device)
        self.train_loader = DataLoader(
            CharCropDataset(mode="train", dataset_dir=cfg.dataset_dir, image_size=cfg.char_image_size, pad_ratio=cfg.bbox_pad_ratio, aug=True),
            batch_size=cfg.char_batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        self.val_loader = DataLoader(
            CharCropDataset(mode="val", dataset_dir=cfg.dataset_dir, image_size=cfg.char_image_size, pad_ratio=cfg.bbox_pad_ratio, aug=False),
            batch_size=cfg.char_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        self.model = CharClassifier(arch=cfg.char_arch, class_num=cfg.char_class_num, pretrained=cfg.pretrained).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.char_lr, weight_decay=cfg.char_weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.char_epochs)
        self.best_char_acc = 0.0
        self.best_path = ""

    def train(self) -> str:
        for epoch in range(self.cfg.char_epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            val_acc = self.eval_char_cls()
            self.scheduler.step()
            print(f"[Char] epoch={epoch + 1}/{self.cfg.char_epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_char_acc={val_acc:.4f}")
            if val_acc > self.best_char_acc:
                self.best_char_acc = val_acc
                Path(self.cfg.checkpoints).mkdir(parents=True, exist_ok=True)
                path = Path(self.cfg.checkpoints) / f"char-{self.cfg.char_arch}-epoch-{epoch + 1}-acc-{val_acc * 100:.2f}.pth"
                torch.save({"model": self.model.state_dict(), "char_arch": self.cfg.char_arch, "char_class_num": self.cfg.char_class_num}, path)
                self.best_path = str(path)
                print(f"Best char checkpoint saved: {self.best_path}")
        return self.best_path

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        corrects = 0
        seen = 0
        tbar = tqdm(self.train_loader, desc=f"Char Epoch {epoch + 1}/{self.cfg.char_epochs}")
        for img, label in tbar:
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            logits = self.model(img)
            loss = self.criterion(logits, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            seen += img.size(0)
            corrects += (pred == label).sum().item()
            tbar.set_postfix(loss=total_loss / max(1, seen / img.size(0)), acc=corrects / seen)
        return total_loss / len(self.train_loader), corrects / seen

    def eval_char_cls(self) -> float:
        self.model.eval()
        corrects = 0
        seen = 0
        with torch.no_grad():
            for img, label in self.val_loader:
                img = img.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                logits = self.model(img)
                pred = logits.argmax(dim=1)
                seen += img.size(0)
                corrects += (pred == label).sum().item()
        self.model.train()
        return corrects / seen


def _load_char_model(model_path: str | Path, cfg: Config = config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    arch = ckpt.get("char_arch", cfg.char_arch)
    class_num = ckpt.get("char_class_num", cfg.char_class_num)
    model = CharClassifier(arch=arch, class_num=class_num, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


def _char_infer_pil(model: nn.Module, device: torch.device, pil_images: list[Image.Image], image_size: int) -> list[int]:
    trans = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    batch = torch.stack([trans(img.convert("RGB")) for img in pil_images]).to(device)
    with torch.no_grad():
        logits = model(batch)
        return logits.argmax(dim=1).detach().cpu().tolist()


def eval_char_bbox(model_path: str | Path, cfg: Config = config, mode: str = "val") -> float:
    paths = data_paths(cfg.dataset_dir)
    labels = load_labels(paths[f"{mode}_label"])
    image_dir = paths[f"{mode}_data"]
    model, device = _load_char_model(model_path, cfg)
    correct = 0
    total = 0
    for file_name, mark in tqdm(labels.items(), desc=f"Eval char bbox ({mode})"):
        img_path = image_dir / file_name
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        boxes = []
        for i, digit in enumerate(mark["label"]):
            bbox = (float(mark["left"][i]), float(mark["top"][i]), float(mark["width"][i]), float(mark["height"][i]))
            boxes.append((bbox, int(digit), float(mark["left"][i])))
        boxes = sorted(boxes, key=lambda x: x[2])
        crops = [crop_digit(image, item[0], cfg.bbox_pad_ratio) for item in boxes]
        preds = _char_infer_pil(model, device, crops, cfg.char_image_size)
        pred_text = "".join(str(d) for d in preds)
        gt_text = "".join(str(item[1]) for item in boxes)
        total += 1
        if pred_text == gt_text:
            correct += 1
    acc = correct / max(total, 1)
    print(f"[Char+BBox] {mode} full-sequence Acc: {acc:.4f}")
    return acc


def export_yolo_dataset(cfg: Config = config) -> Path:
    paths = data_paths(cfg.dataset_dir)
    out_root = Path(cfg.yolo_output_dir)
    for split in ["train", "val"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        labels = load_labels(paths[f"{split}_label"])
        img_dir = paths[f"{split}_data"]
        for file_name, mark in tqdm(labels.items(), desc=f"Export YOLO {split}"):
            src = img_dir / file_name
            if not src.exists():
                continue
            with Image.open(src) as image:
                img_w, img_h = image.size
            lines = []
            for i, digit in enumerate(mark["label"]):
                left = float(mark["left"][i])
                top = float(mark["top"][i])
                width = float(mark["width"][i])
                height = float(mark["height"][i])
                x_center = (left + width / 2.0) / img_w
                y_center = (top + height / 2.0) / img_h
                w = width / img_w
                h = height / img_h
                lines.append(f"{int(digit)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            dst_image = out_root / "images" / split / file_name
            dst_label = out_root / "labels" / split / (Path(file_name).stem + ".txt")
            if not dst_image.exists():
                shutil.copy2(src, dst_image)
            dst_label.write_text("\n".join(lines), encoding="utf-8")

    yaml_path = out_root / "digit.yaml"
    class_names = ", ".join([f"'{i}'" for i in range(10)])
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 10",
                f"names: [{class_names}]",
            ]
        ),
        encoding="utf-8",
    )
    print(f"YOLO dataset exported to: {out_root}")
    print(f"YOLO yaml file: {yaml_path}")
    return yaml_path


def _run_yolo_detect(detector_model_path: str | Path, image_paths: list[Path], conf: float, iou: float):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("未安装 ultralytics，请先运行: pip install ultralytics") from exc
    detector = YOLO(str(detector_model_path))
    return detector.predict(
        source=[str(p) for p in image_paths],
        conf=conf,
        iou=iou,
        verbose=False,
        save=False,
    )


def _decode_detector_only(result) -> str:
    if result.boxes is None or len(result.boxes) == 0:
        return ""
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    order = np.argsort(xyxy[:, 0])
    return "".join(str(cls[i]) for i in order)


def _decode_with_char_model(result, orig_img: np.ndarray, char_model, device: torch.device, cfg: Config) -> str:
    if result.boxes is None or len(result.boxes) == 0:
        return ""
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    order = np.argsort(xyxy[:, 0])
    pil = Image.fromarray(orig_img[..., ::-1]).convert("RGB")
    crops = []
    for idx in order:
        x1, y1, x2, y2 = xyxy[idx]
        bbox = (x1, y1, x2 - x1, y2 - y1)
        crops.append(crop_digit(pil, bbox, cfg.bbox_pad_ratio))
    preds = _char_infer_pil(char_model, device, crops, cfg.char_image_size)
    return "".join(str(d) for d in preds)


def predict_detect(
    detector_model_path: str | Path,
    csv_path: str | Path = config.result_csv,
    cfg: Config = config,
    char_model_path: str = "",
    conf: float | None = None,
    iou: float | None = None,
) -> list[list[str]]:
    conf = cfg.yolo_conf if conf is None else conf
    iou = cfg.yolo_iou if iou is None else iou
    test_images = sorted(data_paths(cfg.dataset_dir)["test_data"].glob("*.png"))
    if not test_images:
        raise FileNotFoundError("测试集图片不存在，请先执行 prepare_dataset。")

    char_model, device = (None, None)
    if char_model_path:
        char_model, device = _load_char_model(char_model_path, cfg)

    results = []
    batch_size = 512
    for start in tqdm(range(0, len(test_images), batch_size), desc="Detect predict"):
        chunk = test_images[start : start + batch_size]
        predicts = _run_yolo_detect(detector_model_path, chunk, conf, iou)
        for img_path, result in zip(chunk, predicts):
            if char_model is not None and device is not None:
                text = _decode_with_char_model(result, result.orig_img, char_model, device, cfg)
            else:
                text = _decode_detector_only(result)
            results.append([img_path.name, text])
    write_result_csv(results, csv_path)
    return results


def parse_predictions(prediction) -> list[str]:
    char_list = [str(i) for i in range(10)] + [""]
    pred_digits = [p.argmax(dim=1).detach().cpu().tolist() for p in prediction]
    return ["".join(char_list[digit] for digit in digits) for digits in zip(*pred_digits)]


def write_result_csv(results: list[list[str]], csv_path: str | Path) -> None:
    df = pd.DataFrame(results, columns=["file_name", "file_code"])
    df["file_name"] = df["file_name"].apply(lambda x: Path(x).name)
    df = df.sort_values("file_name")
    df.to_csv(csv_path, sep=",", index=False)
    print(f"Results saved to {csv_path}")


def predict(
    model_path: str | Path,
    csv_path: str | Path = config.result_csv,
    cfg: Config = config,
) -> list[list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    arch = checkpoint.get("arch", cfg.arch)
    class_num = checkpoint.get("class_num", cfg.class_num)
    max_length = checkpoint.get("max_length", cfg.max_length)
    model = StreetNumberRecognizer(arch=arch, class_num=class_num, max_length=max_length, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_loader = DataLoader(
        DigitsDataset(mode="test", dataset_dir=cfg.dataset_dir, aug=False, max_length=max_length),
        batch_size=cfg.batch_size,
        shuffle=False,
        **_make_loader_kwargs(cfg.num_workers, device),
    )
    results = []
    with torch.no_grad():
        for img, img_names in tqdm(test_loader, desc="Predict"):
            img = img.to(device, non_blocking=True)
            pred = model(img)
            results.extend([[name, code] for name, code in zip(img_names, parse_predictions(pred))])
    write_result_csv(results, csv_path)
    return results


def update_config_from_args(args) -> Config:
    return Config(
        data_list=args.data_list,
        dataset_dir=args.dataset_dir,
        checkpoints=args.checkpoints,
        result_csv=args.result_csv,
        arch=args.arch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        seed=args.seed,
        num_workers=args.num_workers,
        char_arch=args.char_arch,
        char_batch_size=args.char_batch_size,
        char_epochs=args.char_epochs,
        char_lr=args.char_lr,
        char_weight_decay=args.char_weight_decay,
        char_image_size=args.char_image_size,
        bbox_pad_ratio=args.bbox_pad_ratio,
        yolo_output_dir=args.yolo_output_dir,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Street character recognition baseline.")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "prepare",
            "eda",
            "train",
            "predict",
            "all",
            "visualize-char",
            "train-char",
            "eval-char-bbox",
            "export-yolo",
            "predict-detect",
        ],
    )
    parser.add_argument("--data-list", default=config.data_list)
    parser.add_argument("--dataset-dir", default=config.dataset_dir)
    parser.add_argument("--checkpoints", default=config.checkpoints)
    parser.add_argument("--result-csv", default=config.result_csv)
    parser.add_argument("--arch", default=config.arch, choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--detector-model-path", default="")
    parser.add_argument("--char-model-path", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--char-arch", default=config.char_arch, choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--char-batch-size", type=int, default=config.char_batch_size)
    parser.add_argument("--char-epochs", type=int, default=config.char_epochs)
    parser.add_argument("--char-lr", type=float, default=config.char_lr)
    parser.add_argument("--char-weight-decay", type=float, default=config.char_weight_decay)
    parser.add_argument("--char-image-size", type=int, default=config.char_image_size)
    parser.add_argument("--bbox-pad-ratio", type=float, default=config.bbox_pad_ratio)
    parser.add_argument("--yolo-output-dir", default=config.yolo_output_dir)
    parser.add_argument("--yolo-conf", type=float, default=config.yolo_conf)
    parser.add_argument("--yolo-iou", type=float, default=config.yolo_iou)
    parser.add_argument("--vis-count", type=int, default=12)
    parser.add_argument("--eval-mode", default="val", choices=["train", "val"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = update_config_from_args(args)
    set_seed(cfg.seed)

    if args.command in {"prepare", "all"}:
        prepare_dataset(cfg.data_list, cfg.dataset_dir)

    if args.command == "eda" or (args.command == "all" and not args.skip_eda):
        run_eda(cfg.dataset_dir)

    best_model = args.model_path
    if args.command in {"train", "all"}:
        trainer = Trainer(cfg)
        best_model = trainer.train()

    if args.command == "predict" or (args.command == "all" and best_model):
        if not best_model:
            raise ValueError("Please provide --model-path for prediction.")
        predict(best_model, cfg.result_csv, cfg)

    if args.command == "visualize-char":
        visualize_char_crops(cfg.dataset_dir, sample_count=args.vis_count, pad_ratio=cfg.bbox_pad_ratio)

    if args.command == "train-char":
        trainer = CharTrainer(cfg)
        path = trainer.train()
        print(f"[Char] best checkpoint: {path}")

    if args.command == "eval-char-bbox":
        if not args.char_model_path:
            raise ValueError("Please provide --char-model-path for eval-char-bbox.")
        eval_char_bbox(args.char_model_path, cfg, mode=args.eval_mode)

    if args.command == "export-yolo":
        export_yolo_dataset(cfg)

    if args.command == "predict-detect":
        if not args.detector_model_path:
            raise ValueError("Please provide --detector-model-path for predict-detect.")
        predict_detect(
            detector_model_path=args.detector_model_path,
            csv_path=cfg.result_csv,
            cfg=cfg,
            char_model_path=args.char_model_path,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
        )


if __name__ == "__main__":
    main()
