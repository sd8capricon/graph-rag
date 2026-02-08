import shutil
import zipfile
from pathlib import Path

from fastapi import UploadFile


def extract_zip(
    upload: UploadFile,
    temp_dir: Path,
) -> Path:
    """
    Save an uploaded ZIP file and extract it.

    Args:
        upload: FastAPI UploadFile (ZIP)
        temp_dir: Directory to store the ZIP temporarily

    Returns:
        Path to the extraction directory
    """
    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_path = temp_dir / upload.filename

    # Save ZIP
    with zip_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

    # Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir
