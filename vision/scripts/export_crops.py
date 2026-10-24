"""ตัดเมล็ดทีละเม็ดจาก dataset YOLO (polygon หรือกรอบ) เป็น dataset สำหรับ classification (folder ละคลาส)

    .venv/bin/python -I scripts/export_crops.py datasets/robusta-7cls-yolo datasets/robusta-7cls-crops

- ผลลัพธ์: <dst>/{train,val,test}/<class>/*.jpg  (ส่ง <dst> เป็น data= ให้ yolo classify ได้เลย)
- split ตาม dataset ต้นทาง (valid -> val) จึงไม่มีเม็ดจากภาพเดียวกันข้าม split
- crop เป็นสี่เหลี่ยมจัตุรัสรอบเม็ด (center crop ของ yolo-cls จะไม่ตัดปลายเม็ดทิ้ง)
- ถ้า label เป็น polygon พื้นหลังรอบเม็ดถูกระบายเป็นสีเทาเดียวกัน (--keep-bg เพื่อไม่ระบาย)
  ตอน inference ต้อง crop ด้วยวิธีเดียวกัน (mask จาก seg model + พื้นเทา) ให้หน้าตาเหมือนตอน train
"""
import argparse
import collections
from pathlib import Path

import cv2
import numpy as np
import yaml

BG = (128, 128, 128)

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--pad", type=float, default=0.10, help="ขอบเผื่อรอบเม็ด (สัดส่วนของด้านยาว)")
p.add_argument("--keep-bg", action="store_true")
a = p.parse_args()
src, dst = Path(a.src), Path(a.dst)
if dst.exists():
    raise SystemExit(f"{dst} มีอยู่แล้ว ลบหรือใช้ชื่ออื่น (สคริปต์ไม่เขียนทับ)")

names = yaml.safe_load((src / "data.yaml").read_text())["names"]
names = list(names.values()) if isinstance(names, dict) else list(names)

cnt = collections.Counter()
for split, out_split in (("train", "train"), ("valid", "val"), ("test", "test")):
    for n in names:
        (dst / out_split / n).mkdir(parents=True)
    for img_path in sorted((src / split / "images").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lines = (src / split / "labels" / f"{img_path.stem}.txt").read_text().splitlines()
        for i, line in enumerate(l for l in lines if l.strip()):
            c, *v = line.split()
            v = np.array(v, dtype=float)
            if len(v) == 4:
                cx, cy, bw, bh = v
                pts = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                                [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])
                is_poly = False
            else:
                pts, is_poly = v.reshape(-1, 2), True
            pts = (pts * [w, h]).astype(np.int32)
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            side = int(max(x2 - x1, y2 - y1) * (1 + 2 * a.pad))
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            sx, sy = mx - side // 2, my - side // 2

            src_img = img
            if is_poly and not a.keep_bg:
                mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                src_img = np.where(mask[..., None] > 0, img, np.array(BG, np.uint8))
            crop = np.full((side, side, 3), BG, np.uint8)
            ix1, iy1, ix2, iy2 = max(sx, 0), max(sy, 0), min(sx + side, w), min(sy + side, h)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            crop[iy1 - sy:iy2 - sy, ix1 - sx:ix2 - sx] = src_img[iy1:iy2, ix1:ix2]
            name = names[int(c)]
            cv2.imwrite(str(dst / out_split / name / f"{img_path.stem}_{i:02d}.jpg"), crop)
            cnt[out_split, name] += 1

for s in ("train", "val", "test"):
    print(f"{s:5s}", {n: cnt[s, n] for n in names})
print("OK ->", dst)
