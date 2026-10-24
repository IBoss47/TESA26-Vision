# Vision Node: Robusta 7 คลาส

โมเดลภาพสำหรับตรวจเมล็ดกาแฟโรบัสต้า 7 คลาส (`good, shell, full-black, husk, broken-chipped-cut, parchment, immature`)
train ด้วย YOLO26 (Ultralytics) บน Google Colab แล้ว export เป็น NCNN ไปรันบน Raspberry Pi 5

คู่มือนี้พาตั้งแต่ขั้นที่ 1 จนถึง train บน Colab ได้

```
vision/
├── scripts/
│   ├── train_segmentation.py    # ตัวหลัก: mask + กรอบ + คลาสต่อเม็ด
│   ├── train_detection.py       # ตัวเทียบ: กรอบ + คลาสต่อเม็ด
│   ├── train_classification.py  # จำแนกทีละเม็ด (Day 1)
│   └── export_crops.py          # ตัด dataset seg เป็นภาพทีละเม็ดสำหรับ classification
├── colab/train_colab.ipynb      # notebook สำหรับ train บน Colab
├── datasets/robusta-7cls/       # dataset (แตกจาก datasets.zip)
└── datasets.zip                 # ดาวน์โหลดจาก Google Drive (ขั้นที่ 2)
```

---

## ขั้นที่ 1: เตรียมเครื่องตัวเอง (ไม่บังคับ แต่แนะนำ)

ใช้เพื่อลองรันสคริปต์สั้น ๆ ก่อนขึ้น Colab ถ้าจะ train บน Colab อย่างเดียว ข้ามไปขั้นที่ 2 ได้

ต้องมี [uv](https://docs.astral.sh/uv/) (`brew install uv` บน macOS) จากนั้นรันในโฟลเดอร์ `vision/`:

```bash
uv venv --python 3.12
```

```bash
uv pip install ultralytics==8.4.184
```

แตก dataset จากขั้นที่ 2 ไว้ใน `vision/` แล้วลองรัน smoke test (1 epoch, ข้อมูล 20%) เพื่อเช็กว่า dataset และสคริปต์ใช้ได้:

```bash
.venv/bin/python scripts/train_segmentation.py --name smoke_seg --epochs 1 --fraction 0.2
```

ถ้าจบด้วยบรรทัด `TEST box mAP50=... | mask mAP50=...` แปลว่าพร้อม (บน Mac จะใช้ `mps` อัตโนมัติ ช้ากว่า GPU บน Colab มาก)

## ขั้นที่ 2: ดาวน์โหลด dataset

ดาวน์โหลด **[datasets.zip](https://drive.google.com/file/d/1cULKxDoRD_H2sM1wvKrukQ77bqhTGexz/view?usp=drive_link)** จาก Google Drive (~100 MB) มีครบทั้ง segmentation, detection และ classification **ไม่ต้องแตกไฟล์** ไปขั้นที่ 3 ได้เลย

ถ้าจะลองรันในเครื่องตามขั้นที่ 1 ให้แตก zip ในโฟลเดอร์ `vision/` จะได้ `vision/datasets/robusta-7cls/...`

## ขั้นที่ 3: อัปโหลดขึ้น Google Drive

สร้างโฟลเดอร์ใน Drive แล้วอัปโหลดให้ได้โครงนี้:

```
MyDrive/TESA/vision/
├── scripts/        ← ทั้งโฟลเดอร์ vision/scripts
└── datasets.zip    ← ไฟล์จากขั้นที่ 2 (ห้ามแตก zip และห้ามเปลี่ยนชื่อ)
```

ถ้าใช้ชื่อโฟลเดอร์อื่น ให้ไปแก้ `DRIVE_DIR` ใน notebook (ขั้นที่ 5)
ถ้าแก้สคริปต์ในเครื่อง ต้องอัปโหลด `scripts/` ทับใน Drive ใหม่ทุกครั้ง

## ขั้นที่ 4: เปิด notebook และเลือก GPU

1. เข้า [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → เลือก `colab/train_colab.ipynb`
2. **Runtime → Change runtime type → T4 GPU** → Save
3. รัน cell แรก (`!nvidia-smi`) ต้องเห็นชื่อการ์ด Tesla T4 ถ้าขึ้น error แปลว่ายังไม่ได้เลือก GPU

## ขั้นที่ 5: รัน notebook ทีละ cell

| cell | ทำอะไร | หมายเหตุ |
|---|---|---|
| `drive.mount` | เชื่อม Google Drive | กดอนุญาตบัญชีที่มีไฟล์จากขั้นที่ 3 |
| ตั้งค่า | `DRIVE_DIR`, `RUNS_DIR` | แก้ถ้าใช้โฟลเดอร์อื่น |
| เตรียมโค้ด + dataset | copy scripts, แตก zip ไป `/content/vision`, ติดตั้ง ultralytics | **รันใหม่ทุกครั้งที่ runtime ถูก reset** |
| 1) Smoke test | train 1 epoch | ต้องจบด้วยบรรทัด `TEST ...` ก่อนไปต่อ |
| 2) Segmentation | train ตัวหลัก 150 epoch + export NCNN | ประมาณ 20–30 นาทีบน T4 |
| 3) Detection | ตัวเทียบ | ไม่บังคับ |
| 4) Classification | จำแนกทีละเม็ด | สำหรับ Day 1 |
| ดูผล | แสดง `results.png` และ confusion matrix | แก้ `RUN` เป็นชื่อ run ที่จะดู |

### ตัวเลือกที่ใช้บ่อย (ต่อท้ายคำสั่ง train)

| ตัวเลือก | ค่าเริ่มต้น | ใช้เมื่อ |
|---|---|---|
| `--name` | (ต้องใส่) | ชื่อ run ควรไม่ซ้ำ ให้เพิ่มเลขเวอร์ชันทุกครั้ง เช่น `seg_y26n_v2` (ถ้าซ้ำ ผลจะไปอยู่ที่ `<name>-2`) |
| `--model` | `yolo26n-seg.pt` | `yolo26s-seg.pt` แม่นขึ้นแต่ช้าลงบน Pi |
| `--epochs` | 150 (cls 100) | ลดลงเพื่อดูผลเร็ว / เพิ่มถ้ากราฟยังขึ้นตอนจบ |
| `--batch` | 16 (cls 64) | เพิ่มได้ถ้า GPU memory เหลือ |
| `--workers` | 4 | **ใส่ `2` บน Colab** (มี CPU 2 ตัว) |
| `--export` | `none` | `ncnn` สำหรับ Pi, `onnx` สำหรับทดสอบทั่วไป |
| `--project` | `vision/runs` | notebook ชี้ไป Drive ให้แล้ว ผลไม่หายเมื่อ Colab หลุด |

## ขั้นที่ 6: ผลลัพธ์อยู่ไหน และอ่านอย่างไร

ทุก run อยู่ใน Drive ที่ `MyDrive/TESA/vision/runs/`:

```
runs/<name>/
├── weights/best.pt              # โมเดลที่ดีที่สุด (ใช้ต่อ / เทียบผล)
├── weights/last.pt              # ใช้ train ต่อเมื่อ Colab หลุด
├── weights/best_ncnn_model/     # ไฟล์สำหรับ Raspberry Pi (ถ้าใส่ --export ncnn)
├── results.csv / results.png    # loss และ mAP ราย epoch
runs/<name>_test/                # ผลบนชุด test + confusion matrix
```

- **seg / det:** ดู `mAP50` และ `mAP50-95` ทั้ง Box (B) และ Mask (M) ยิ่งใกล้ 1 ยิ่งดี
- **cls:** ดู `top1`
- ชุด val มีภาพน้อย ค่า mAP ราย epoch จึงแกว่ง ให้ดูแนวโน้มและใช้ผลบนชุด **test** ตัดสิน
- ถ้า `box_loss`/`seg_loss` ต่ำแต่ mAP ไม่ขึ้น ปัญหามักอยู่ที่การแยกคลาส ให้ดู confusion matrix ว่าคู่ไหนสับสน

## แก้ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| หาโฟลเดอร์ผลไม่เจอ / มีโฟลเดอร์ `<name>-2` | ใช้ `--name` ซ้ำ ultralytics จึงสร้างโฟลเดอร์ใหม่แทนการทับ ดูชื่อจริงจากบรรทัด `best weights:` ท้าย log |
| `cp: cannot stat ... scripts` หรือ `unzip: cannot find` | `DRIVE_DIR` ไม่ตรงกับโฟลเดอร์ใน Drive หรือยังไม่ได้ mount |
| `Dataset ... images not found` | ไม่ได้ใช้ `datasets.zip` จาก link ในขั้นที่ 2 หรือแตก zip / เปลี่ยนชื่อไฟล์ก่อนอัปโหลด |
| train ช้ามาก / ขึ้น `device=cpu` | ยังไม่ได้เลือก T4 GPU (ขั้นที่ 4) |
| Colab หลุดกลางทาง | รัน cell เตรียมโค้ดใหม่ แล้วใช้ cell "Colab หลุด" ท้าย notebook (`resume=True` จาก `last.pt`) จากนั้นรัน cell test/export |
| `ModuleNotFoundError: ultralytics` | runtime ถูก reset ให้รัน cell เตรียมโค้ดอีกครั้ง |
