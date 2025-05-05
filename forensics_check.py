import os
import hashlib
from PIL import Image, ImageChops, ImageEnhance
import io
import subprocess

def extract_exif(image_path):
    try:
        result = subprocess.run(['exiftool', image_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return None
        
        exif_raw = result.stdout.strip().split('\n')
        exif_data = {}

        for line in exif_raw:
            if ':' in line:
                key, value = line.split(':', 1)
                exif_data[key.strip()] = value.strip()
        return exif_data
    except Exception as e:
        print(f"EXIF extraction error: {e}")
        return None


def compute_hashes(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
        return {
            "MD5": hashlib.md5(data).hexdigest(),
            "SHA256": hashlib.sha256(data).hexdigest(),
            "SHA512": hashlib.sha512(data).hexdigest()
        }

def perform_ela(image_path, quality=95):
    try:
        original = Image.open(image_path).convert('RGB')
        buffer = io.BytesIO()
        original.save(buffer, format='JPEG', quality=quality)
        ela_image = Image.open(buffer)
        diff = ImageChops.difference(original, ela_image)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        diff = ImageEnhance.Brightness(diff).enhance(scale)

        ela_path = os.path.join("static/results", "ela_" + os.path.basename(image_path))
        diff.save(ela_path)
        return ela_path
    except Exception as e:
        return None
