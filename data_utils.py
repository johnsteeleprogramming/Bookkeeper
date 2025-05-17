import os
import textwrap


def clean_code(code: str) -> str:
    if "```" in code:
        parts = code.split("```")
        code = next((p for p in parts if "import" in p or "fig" in p), code)
        if code.strip().startswith("python"):
            code = code.strip()[6:]
    return textwrap.dedent(code.strip())


def save_uploaded_file(file, upload_folder) -> str:
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)
    return filepath
