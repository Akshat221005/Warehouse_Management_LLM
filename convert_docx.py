from docx import Document
import os

path = r"C:\Users\aksha\OneDrive\Desktop\Warehouse_Management_Help_2021.docx"

print("Checking file exists:", os.path.exists(path))

doc = Document(path)

texts = []
for para in doc.paragraphs:
    t = para.text.strip()
    if len(t) > 0:
        texts.append(t)

full_text = "\n".join(texts)

os.makedirs("data", exist_ok=True)

with open("data/wms.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Saved data/wms.txt")
