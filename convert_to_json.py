import json
import re
import os


def text_to_legal_json(input_file, output_file):
    corpus_data = []
    current_category = "General"  # ค่าเริ่มต้น
    current_source = "พ.ร.บ. ทรัพย์สินทางปัญญา"

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

        for i, text in enumerate(lines):
            # 1. ถ้าเจอเครื่องหมาย # ให้เปลี่ยน Category ตามหัวข้อนั้นๆ
            if text.startswith("#"):
                if "สิทธิบัตร" in text:
                    current_category = "Patent"
                elif "ลิขสิทธิ์" in text:
                    current_category = "Copyright"
                elif "เครื่องหมายการค้า" in text:
                    current_category = "Trademark"
                else:
                    current_category = "General"
                current_source = text.replace("#", "").strip("- ")
                continue

            # 2. ใช้ Regex ดึงเลขมาตรา
            section_match = re.search(r'มาตรา\s*([๐-๙\d]+)', text)
            section_no = section_match.group(
                1) if section_match else f"line_{i}"

            entry = {
                "id": f"{current_category}_{section_no}",
                "category": current_category,  # ใช้ค่าที่จำมาจากหัวข้อล่าสุด
                "section": section_no,
                "text": text,
                "source": current_source
            }
            corpus_data.append(entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=4)

    print(f"✅ แปลงไฟล์สำเร็จ! ได้ข้อมูลคุณภาพ {len(corpus_data)} รายการ")


# เรียกใช้งาน (Path ตามที่พี่วางไว้)
text_to_legal_json('data/raw/thai_ip_corpus.txt',
                   'data/processed/thai_ip_corpus.json')
