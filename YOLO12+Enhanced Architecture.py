!pip install ultralytics
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
from pathlib import Path
class GCBlock(nn.Module):
    """Global Context Block"""
    def __init__(self, in_channels):
        super(GCBlock, self).__init__()
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.LayerNorm([in_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        mask = self.softmax(self.conv_mask(x).view(B, 1, -1))
        context = torch.sum(x.view(B, C, -1) * mask, dim=2).view(B, C, 1, 1)
        return x + self.transform(context)

class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        ch_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * ch_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

        return x * sp_att
class HybridAttention(nn.Module):
    """GC + ECA + CBAM"""
    def __init__(self, in_channels):
        super(HybridAttention, self).__init__()
        self.gc = GCBlock(in_channels)
        self.eca = ECA(in_channels)
        self.cbam = CBAM(in_channels)

    def forward(self, x):
        x = self.gc(x)
        x = self.eca(x)
        x = self.cbam(x)
        return x
class YOLOv12_Enhanced(nn.Module):
    def __init__(self, model_size='x'):
        super(YOLOv12_Enhanced, self).__init__()
        model_name = f"yolo12{model_size}.pt"
        base = YOLO(model_name)
        self.model = base.model

        modules = list(self.model.model.children())
        new_layers = []
        attention_count = 0

        for layer in modules:
            new_layers.append(layer)

            if layer.__class__.__name__.startswith("C2f"):
                try:
                    out_ch = None
                    if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
                        if hasattr(layer.cv2.conv, 'conv'):
                            out_ch = layer.cv2.conv.conv.out_channels
                        elif hasattr(layer.cv2.conv, 'out_channels'):
                            out_ch = layer.cv2.conv.out_channels
                    elif hasattr(layer, 'c2') and hasattr(layer.c2, 'out_channels'):
                        out_ch = layer.c2.out_channels

                    if out_ch:
                        new_layers.append(HybridAttention(out_ch))
                        attention_count += 1
                except:
                    pass

        self.model.model = nn.Sequential(*new_layers)
        print(f"Added {attention_count} attention blocks")

    def forward(self, x):
        return self.model(x)
def calculate_class_weights(data_dir):
    labels_dir = Path(data_dir) / "labels" / "train"

    if not labels_dir.exists():
        print("Labels not found → Using default weights")
        return None

    class_counts = np.zeros(9)
    total = 0

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if 0 <= cid < 9:
                        class_counts[cid] += 1
                        total += 1

    if total > 0:
        weights = total / (class_counts + 1e-6)
        weights = weights / weights.sum() * 9
        return weights.tolist()

    return None
def train_yolov12(
    data_dir,
    epochs=200,
    batch=32,
    img=640,
    model_size='x',
    device='0',
    use_class_weights=False
):
    os.makedirs("meta", exist_ok=True)

    yaml_path = "meta/data.yaml"

    with open(yaml_path, "w") as f:
        f.write(f"""
train: '{data_dir}/images/train'
val: '{data_dir}/images/val'
test: '{data_dir}/images/test'

nc: 9
names:
  0: boneanomaly
  1: bonelesion
  2: foreignbody
  3: fracture
  4: metal
  5: periostealreaction
  6: pronatorsign
  7: softtissue
  8: text
""")

    class_weights = calculate_class_weights(data_dir) if use_class_weights else None

    enhanced = YOLOv12_Enhanced(model_size=model_size)
    torch.save(enhanced.state_dict(), "enhanced_model.pt")

    model = YOLO(f"yolo12{model_size}.pt")
    model.model.load_state_dict(enhanced.state_dict(), strict=False)

    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch,
        imgsz=img,
        device=device,
        name=f"YOLOv12_{model_size}_Enhanced"
    )
train_yolov12(
    data_dir="/content/dataset",
    epochs=200,
    batch=32,
    img=640,
    model_size="x",
    device="0",
    use_class_weights=True
)
