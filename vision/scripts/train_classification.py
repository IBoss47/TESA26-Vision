"""Day 1: train YOLO26-cls จำแนก 7 คลาสจากภาพเมล็ดทีละเม็ด (รันได้ทั้งเครื่องนี้และ Colab)

    python scripts/train_classification.py --name cls_y26n_v1
    python scripts/train_classification.py --name smoke_cls --epochs 1          # ทดสอบ pipeline

dataset: datasets/robusta-7cls/classification/{train,val,test}/<class>/*.jpg (สร้างด้วย export_crops.py)
ตอน inference ต้อง crop แบบเดียวกัน: จัตุรัสรอบเม็ด + พื้นหลังสีเทา (128) แล้วโหวตทุกเม็ดเป็นคลาสของภาพ
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # vision/

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="ชื่อ run (ผลอยู่ที่ <project>/<name>)")
p.add_argument("--data", default=str(ROOT / "datasets/robusta-7cls/classification"))
p.add_argument("--model", default="yolo26n-cls.pt")
p.add_argument("--imgsz", type=int, default=224)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--patience", type=int, default=30)
p.add_argument("--device", default="auto", help="auto | 0 | mps | cpu")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", default=str(ROOT / "runs"), help="Colab: ชี้ไป Google Drive กันผลหาย")
p.add_argument("--export", choices=["none", "onnx", "ncnn"], default="none")
a = p.parse_args()

device = a.device
if device == "auto":
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
data = str(Path(a.data).resolve())

model = YOLO(a.model)  # เริ่มจาก pretrained เสมอ
model.train(
    data=data,
    imgsz=a.imgsz,
    epochs=a.epochs,
    patience=a.patience,
    batch=a.batch,
    device=device,
    workers=a.workers,
    seed=a.seed,
    project=a.project,
    name=a.name,
    exist_ok=False,  # ชื่อซ้ำจะ error แทนการทับผลเก่า
    # สีคือสิ่งที่แยก good / immature / parchment: ไม่เลื่อน hue, ปิด randaugment (มีการปรับสีแรง)
    hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, auto_augment="", erasing=0.2,
    fliplr=0.5, flipud=0.5,  # เมล็ดไม่มีทิศบน-ล่าง
)

best = Path(model.trainer.save_dir) / "weights" / "best.pt"
m = YOLO(best)
r = m.val(data=data, imgsz=a.imgsz, split="test", device=device, plots=True, verbose=False,
          project=a.project, name=f"{a.name}_test")
print(f"TEST top1={r.top1:.4f} top5={r.top5:.4f}")
if a.export != "none":
    print("exported:", m.export(format=a.export, imgsz=a.imgsz))
print("best weights:", best)
