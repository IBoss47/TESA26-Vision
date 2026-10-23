# Robusta Lot Passport (TGR2026)

ติดตามคุณภาพกาแฟโรบัสตาทีละล็อตด้วย `lot_id` ตั้งแต่ชั่งผลสด ตรวจตาม มกษ. 5700 คิดคะแนน Q และราคา แล้วออก QR Passport

## โครงสร้าง
| โฟลเดอร์ | งาน | เจ้าของ |
|---|---|---|
| firmware/ | ESP32 เซ็นเซอร์ | คนที่ 1 |
| vision/ | YOLOv8n บน Pi 5 | คนที่ 2 |
| edge/ | Edge Backend + SQLite | คนที่ 3 |
| market/ | Q score ราคา QR | คนที่ 4 |
| dashboard/ | เว็บมือถือ + LLM | คนที่ 5 |
| contracts/ | MQTT topic และ API schema | ทุกคน |
| docs/ | รายงาน รูป คู่มือ label | ทุกคน |

## วิธีรัน
แต่ละโฟลเดอร์มี README ของตัวเองว่าติดตั้งและรันยังไง
ลำดับเปิดระบบตอนเดโม: MQTT broker บน Pi 5 → edge → firmware/vision → market กับ dashboard บน Notebook

## เข้าร่วม repo
ทุกคนเป็น collaborator สิทธิ์ Write (ไม่ต้อง fork) รับคำเชิญทางอีเมลก่อนเริ่มงาน

## วิธีทำงานกับ branch
1. `main` ต้องรันได้เสมอ
2. 1 feature = 1 branch แตกจาก main ตั้งชื่อ `<โฟลเดอร์>-<งาน>` เช่น `vision-tray-count`
3. แก้เฉพาะโฟลเดอร์ของตัวเอง (ยกเว้น `contracts/` กับ `docs/` ที่ทุกคนแก้ได้) ทำ feature ให้เล็ก เสร็จภายในไม่กี่ชั่วโมง

    git checkout main && git pull
    git checkout -b vision-tray-count
    # ทำงาน แล้ว commit
    git push -u origin vision-tray-count

4. เปิด PR เข้า main ให้อีก 1 คนดู (ควรเป็นคนสำรองของส่วนนั้น)
5. merge แล้วลบ branch ทิ้ง
6. แก้ `contracts/` ต้องบอกในกลุ่มก่อน และให้เจ้าของส่วนที่กระทบดู PR
7. ห้าม commit รหัสผ่าน .env dataset หรือไฟล์โมเดล ให้เก็บที่ Roboflow/Drive: <ใส่ลิงก์>

## เวอร์ชัน (tag)
ใช้เลข 3 ชุด `vMAJOR.MINOR.PATCH` และสร้างแบบ annotated tag (`-a -m`) เสมอ
- `v0.x.0` ระบบยังต่อไม่ครบ เช่น `v0.1.0` ESP32 ส่งข้อมูลถึง Edge Backend
- `v1.0.0` รันครบ end-to-end ครั้งแรก
- `v1.0.1` แก้บั๊กเล็ก
- `v1.1.0` เพิ่ม feature เช่นงานโจทย์กะทันหัน

    git tag -a v1.0.0 -m "ระบบครบ end-to-end พร้อมเดโม"
    git push origin v1.0.0

## ช่วงแข่ง
- แก้เล็กๆ ในโฟลเดอร์ตัวเอง push เข้า main ตรงได้ แต่ต้อง `git pull` และรันส่วนของตัวเองให้ผ่านก่อน
- ก่อนเดโมติด tag เวอร์ชันล่าสุดที่รันผ่านไว้เสมอ ถ้าพังให้ย้อนกลับไปที่ tag นั้น