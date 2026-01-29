# 📂 Metadata Extractor

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)

Extract detailed metadata from images, PDFs, audio, and video files. Simple, safe, and offline.

## ✨ Features

- 🖼️ **Images** - EXIF data, GPS coordinates, camera settings
- 📄 **PDFs** - Document properties, author, dates
- 🎵 **Audio** - ID3 tags, duration, bitrate
- 🎬 **Video** - Codecs, resolution, embedded metadata
- 🔐 **Privacy** - Optional redaction of sensitive data
- 🔒 **Safe** - Read-only, no execution, completely offline

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/metadata-extractor.git
cd metadata-extractor
pip install -r requirements.txt

# Basic usage
python metadata_extractor.py photo.jpg

# JSON output
python metadata_extractor.py document.pdf json

# Redact sensitive info (GPS, serial numbers, etc.)
python metadata_extractor.py photo.jpg --redact
```

## 📋 Supported Files

| Type | Formats |
|------|---------|
| Images | JPG, PNG, TIFF, GIF, BMP |
| Documents | PDF |
| Audio | MP3, FLAC, WAV, OGG |
| Video | MP4, AVI, MKV, WebM |

## 💻 Usage

**Command Line:**
```bash
python metadata_extractor.py <file_path> [json] [--redact]
```

**Python API:**
```python
from metadata_extractor import extract_metadata_from_file

# Extract metadata
result = extract_metadata_from_file('photo.jpg')
print(result)

# JSON format with redaction
result = extract_metadata_from_file('photo.jpg', 'json', redact_sensitive=True)
```

## 📦 Installation

**Requirements:**
- Python 3.7+
- FFmpeg (optional, for video files)

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Or manually:**
```bash
pip install Pillow PyPDF2 mutagen ffmpeg-python
```

## 🔐 Privacy

Use `--redact` to automatically hide:
- GPS coordinates
- Serial numbers
- Names and personal info
- Email addresses

## 📖 Example Output

```
================================================================================
METADATA EXTRACTION REPORT
================================================================================

[BASIC]
  FileName       : photo.jpg
  FileSize       : 3.88 MB
  MD5Hash        : c576c25917abef2727c85adec5bab5d5

[IMAGE]
  ImageSize      : 4080x3060
  Megapixels     : 12.48
  
  [EXIF]
    Make         : Canon
    Model        : Canon EOS 5D Mark IV
    ISO          : 400
    Aperture     : f/2.8
    ShutterSpeed : 1/125
    
    [GPS]
      Latitude   : 37.774833
      Longitude  : -122.419667
```

## ⚠️ Troubleshooting

**Missing library errors:**
```bash
pip install Pillow PyPDF2 mutagen ffmpeg-python
```

**Video extraction fails:**
Install FFmpeg for your operating system.

**No GPS data in photos:**
Social media apps (WhatsApp, Instagram) strip metadata. Use original photos from your camera/phone.


## 🤝 Contributing

Pull requests are welcome! Feel free to open issues for bugs or feature requests.

---

⭐ **Star this repo if you find it useful!**
