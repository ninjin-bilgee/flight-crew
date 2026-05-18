import os
import uuid

pdf_folder = "resumes_pdf"

for filename in os.listdir(pdf_folder):

    if filename.lower().endswith(".pdf"):

        base_name = os.path.splitext(filename)[0]

        unique_id = str(uuid.uuid4())[:8]

        new_name = f"{base_name}_{unique_id}.pdf"

        old_path = os.path.join(pdf_folder, filename)
        new_path = os.path.join(pdf_folder, new_name)

        os.rename(old_path, new_path)

        print(f"Renamed: {filename} -> {new_name}")