from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import requests


image = Path("inputs/apartment.webp")
with image.open("rb") as file:
    response = requests.post(
        "http://localhost:8000/extract",
        params={"preset": "weak"},
        files={"image": (image.name, file)},
        timeout=300,
    )
response.raise_for_status()
with ZipFile(BytesIO(response.content)) as archive:
    archive.extractall("outputs/apartment")
