#yolo12+GC block on grazpedwri-dx dataset for large model
#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO

# =============================
#   Global Context (GC) Block
# =============================
class GCBlock(nn.Module):
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

# =============================
#   YOLOv12-GC Architecture
# =============================
class YOLOv12_GC(nn.Module):
    def __init__(self):
        super(YOLOv12_GC, self).__init__()
        base = YOLO("yolo12l.pt")
        self.model = base.model

        modules = list(self.model.model.children())
        new_layers = []

        for layer in modules:
            new_layers.append(layer)
            if layer.__class__.__name__.startswith("C2f"):
                try:
                    out_ch = layer.cv2.conv.conv.out_channels
                    new_layers.append(GCBlock(out_ch))
                except:
                    pass

        self.model.model = nn.Sequential(*new_layers)

    def forward(self, x):
        return self.model(x)


# =============================
#   MAIN: CLI ENTRY POINT
# =============================
def main():
    parser = argparse.ArgumentParser(description="Train YOLOv12-L with GC Attention")

    parser.add_argument('--data-dir', type=str, required=True,
                        help='Dataset directory (root folder containing images/train, images/valid, etc.)')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')

    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size')

    parser.add_argument('--device', type=str, default='0',
                        help='Device (0,1 or cpu)')

    args = parser.parse_args()

    # =========================
    # Create data.yaml file
    # =========================
    yaml_text = f"""
train: '{args.data_dir}/images/train'
val: '{args.data_dir}/images/valid'
test: '{args.data_dir}/images/test'

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
"""

    with open("data.yaml", "w") as f:
        f.write(yaml_text)

    print("📁 data.yaml created successfully!")

    # =========================
    # Build GC Model
    # =========================
    print("🔧 Building YOLOv12-L + GC...")
    gc_model = YOLOv12_GC()
    torch.save(gc_model.state_dict(), "gc_model_weights.pt")

    # =========================
    # Load main YOLO model
    # =========================
    print("🚀 Loading model for training...")
    model = YOLO("yolo12l.pt")
    model.model.load_state_dict(gc_model.state_dict(), strict=False)

    # =========================
    # Training
    # =========================
    print("🔥 Starting training...")
    model.train(
        data="data.yaml",
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        name="YOLOv12_Large_GC_CLI",
    )

    # =========================
    # Validation
    # =========================
    print("📊 Running validation...")
    results = model.val(data="data.yaml", imgsz=args.img_size, device=args.device)

    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")

    precision = results.box.p.mean()
    recall = results.box.r.mean()

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    print(f"F1 Score: {f1:.3f}")


# RUN SCRIPT
if __name__ == "__main__":
    main()
