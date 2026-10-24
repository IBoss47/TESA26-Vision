"""Day 2: train YOLO26 detection (กรอบ + คลาสต่อเม็ด) 7 คลาส (รันได้ทั้งเครื่องนี้และ Colab)

    python scripts/train_detection.py --name det_y26n_v1
    python scripts/train_detection.py --name smoke_det --epochs 1 --fraction 0.2   # ทดสอบ pipeline

dataset: datasets/robusta-7cls/detection (label กรอบ สร้างจาก polygon ด้วย make_detection_dataset.py)
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # vision/

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="ชื่อ run (ผลอยู่ที่ <project>/<name>)")
p.add_argument("--data", default=str(ROOT / "datasets/robusta-7cls/detection/data.yaml"))
p.add_argument("--model", default="yolo26n.pt")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--patience", type=int, default=40)
p.add_argument("--device", default="auto", help="auto | 0 | mps | cpu")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--fraction", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", default=str(ROOT / "runs"), help="Colab: ชี้ไป Google Drive กันผลหาย")
p.add_argument("--export", choices=["none", "onnx", "ncnn"], default="none")
a = p.parse_args()

device = a.device
if device == "auto":
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
data = str(Path(a.data).resolve())  # path เต็ม ให้ ultralytics หา root ของ dataset ถูก

model = YOLO(a.model)  # เริ่มจาก pretrained เสมอ
model.train(
    data=data,
    imgsz=a.imgsz,
    epochs=a.epochs,
    patience=a.patience,
    batch=a.batch,
    device=device,
    workers=a.workers,
    fraction=a.fraction,
    seed=a.seed,
    project=a.project,
    name=a.name,
    exist_ok=False,  # ชื่อซ้ำจะ error แทนการทับผลเก่า
    hsv_h=0.0, hsv_s=0.3, hsv_v=0.3,  # สีใช้แยกคลาส: ไม่เลื่อน hue
    fliplr=0.5, flipud=0.5,  # พลิกได้ทุกทาง; ไม่หมุนองศาอิสระเพราะกรอบจะหลวม
)

best = Path(model.trainer.save_dir) / "weights" / "best.pt"
m = YOLO(best)
r = m.val(data=data, imgsz=a.imgsz, split="test", device=device, plots=True, verbose=False,
          project=a.project, name=f"{a.name}_test")
print(f"TEST box mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}")
if a.export != "none":
    print("exported:", m.export(format=a.export, imgsz=a.imgsz))
print("best weights:", best)
