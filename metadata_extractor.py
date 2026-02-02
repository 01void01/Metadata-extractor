"""
Comprehensive Metadata Extractor
Extracts detailed metadata from PDF, Images, Audio, and Video files
Supports JSON and human-readable output formats with optional redaction
Safe read-only analysis - does not execute any file content
"""

import os
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union
import hashlib
import binascii

# Core libraries
try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Image metadata extraction will be limited.")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not installed. PDF metadata extraction disabled.")

try:
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    print("Warning: Mutagen not installed. Audio metadata extraction disabled.")

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print("Warning: ffmpeg-python not installed. Video metadata extraction will be limited.")


class MetadataExtractor:
    """Main class for extracting metadata from various file types"""
    
    # Sensitive fields that can be redacted
    SENSITIVE_FIELDS = {
        'gps', 'gpsinfo', 'location', 'latitude', 'longitude', 'gpslatitude', 
        'gpslongitude', 'gpsaltitude', 'coordinates', 'serialnumber', 
        'cameraserialnumber', 'lensserialnumber', 'ownername', 'artist', 
        'copyright', 'author', 'creator', 'publisher', 'username', 'user',
        'company', 'manager', 'subject', 'keywords', 'comments', 'description',
        'personinimage', 'creatortool', 'macaddress', 'email', 'phone',
        'internalserializenumber', 'cameraownername'
    }
    
    def __init__(self, filepath: str, redact_sensitive: bool = False):
        self.filepath = Path(filepath)
        self.metadata = {}
        self.redact_sensitive = redact_sensitive
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
    
    def extract_all(self) -> Dict[str, Any]:
        """Extract all available metadata from the file"""
        # Basic file metadata (always available)
        self.metadata['basic'] = self._extract_basic_metadata()
        
        # File type specific metadata
        mime_type = self._get_mime_type()
        
        if mime_type:
            if mime_type.startswith('image/'):
                self.metadata['image'] = self._extract_image_metadata()
            elif mime_type == 'application/pdf':
                self.metadata['pdf'] = self._extract_pdf_metadata()
            elif mime_type.startswith('audio/'):
                self.metadata['audio'] = self._extract_audio_metadata()
            elif mime_type.startswith('video/'):
                self.metadata['video'] = self._extract_video_metadata()
        
        # Apply redaction if requested
        if self.redact_sensitive:
            self.metadata = self._redact_metadata(self.metadata)
        
        return self.metadata
    
    def _extract_basic_metadata(self) -> Dict[str, Any]:
        """Extract basic file system metadata"""
        stat = self.filepath.stat()
        
        basic = {
            'FileName': self.filepath.name,
            'FilePath': str(self.filepath.absolute()),
            'FileSize': stat.st_size,
            'FileSizeHuman': self._human_readable_size(stat.st_size),
            'FileType': self.filepath.suffix.upper().replace('.', '') if self.filepath.suffix else 'Unknown',
            'FileExtension': self.filepath.suffix,
            'MIMEType': self._get_mime_type(),
            'FileAccessDate': datetime.fromtimestamp(stat.st_atime).strftime('%Y-%m-%d %H:%M:%S'),
            'FileModifyDate': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'FileInodeChangeDate': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'FilePermissions': oct(stat.st_mode)[-4:],
            'FilePermissionsReadable': self._get_permission_string(stat.st_mode),
            'MD5Hash': self._calculate_hash('md5'),
            'SHA1Hash': self._calculate_hash('sha1'),
            'SHA256Hash': self._calculate_hash('sha256'),
        }
        
        # Add inode and hard link count on Unix systems
        try:
            basic['Inode'] = stat.st_ino
            basic['HardLinkCount'] = stat.st_nlink
            basic['UID'] = stat.st_uid
            basic['GID'] = stat.st_gid
        except AttributeError:
            pass
        
        return basic
    
    def _extract_image_metadata(self) -> Dict[str, Any]:
        """Extract metadata from image files"""
        if not PIL_AVAILABLE:
            return {'error': 'Pillow library not available'}
        
        try:
            with Image.open(self.filepath) as img:
                metadata = {
                    'ImageWidth': img.width,
                    'ImageHeight': img.height,
                    'ImageSize': f"{img.width}x{img.height}",
                    'ColorType': img.mode,
                    'Format': img.format,
                    'Megapixels': round((img.width * img.height) / 1_000_000, 2),
                    'BitsPerPixel': len(img.getbands()) * 8 if hasattr(img, 'getbands') else None,
                    'HasAlphaChannel': img.mode in ('RGBA', 'LA', 'PA'),
                    'IsAnimated': getattr(img, 'is_animated', False),
                    'FrameCount': getattr(img, 'n_frames', 1),
                }
                
                # Extract EXIF data with full IFD parsing
                exif_data = img.getexif()
                if exif_data:
                    exif = {}
                    processed_tags = set()  # Track processed tags to avoid duplicates
                    
                    # Get all EXIF data including all IFDs
                    all_exif_data = dict(exif_data)
                    
                    # Get ExifIFD (detailed camera settings)
                    try:
                        exif_ifd = exif_data.get_ifd(0x8769)
                        all_exif_data.update(exif_ifd)
                    except (KeyError, AttributeError):
                        pass
                    
                    # Process all EXIF tags
                    for tag_id, value in all_exif_data.items():
                        tag = TAGS.get(tag_id, f"Unknown_Tag_{tag_id}")
                        
                        # Skip pointers/offsets as we handle them separately
                        if tag in ['ExifOffset', 'GPSInfo', 'ExifInteroperabilityOffset']:
                            continue
                        
                        # Skip if already processed (avoids duplicates)
                        if tag in processed_tags:
                            continue
                        processed_tags.add(tag)
                        
                        # Skip unknown manufacturer-specific tags
                        if tag.startswith('Unknown_Tag_') or tag.startswith('Tag_'):
                            continue
                        
                        # Skip empty or whitespace-only values
                        if isinstance(value, str) and not value.strip():
                            continue
                        
                        # Convert bytes to string for JSON serialization
                        if isinstance(value, bytes):
                            try:
                                decoded = value.decode('utf-8', errors='ignore').strip()
                                if decoded:  # Only add if not empty after decoding
                                    value = decoded
                                else:
                                    continue
                            except:
                                value = binascii.hexlify(value).decode('ascii')
                        # Handle rational numbers (common in EXIF for exposure, aperture, etc.)
                        elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                            try:
                                if value.denominator == 0:
                                    value = 0
                                else:
                                    # Keep as fraction for shutter speed, convert to float for others
                                    float_value = float(value)
                                    if tag == 'ExposureTime' and float_value < 1:
                                        value = f"1/{int(1/float_value)}"  # Show as "1/60" etc.
                                    else:
                                        value = round(float_value, 3)
                            except:
                                value = str(value)
                        # Handle tuples and lists
                        elif isinstance(value, (list, tuple)):
                            try:
                                value = list(value)
                                # Skip empty lists
                                if not value:
                                    continue
                            except:
                                value = str(value)
                        
                        # Clean up UserComment (often contains debug data)
                        if tag == 'UserComment' and isinstance(value, str):
                            # Truncate if too long or messy
                            if len(value) > 200 or '\n' in value:
                                value = value[:200] + '...' if len(value) > 200 else value
                                value = value.replace('\n', ' ')
                        
                        exif[tag] = value
                    
                    # Handle GPS data separately
                    try:
                        gps_ifd = exif_data.get_ifd(0x8825)
                        if gps_ifd:
                            gps_data = {}
                            for gps_tag_id, gps_value in gps_ifd.items():
                                gps_tag = GPSTAGS.get(gps_tag_id, f"GPSTag_{gps_tag_id}")
                                
                                # Convert GPS coordinate tuples to readable format
                                if gps_tag in ['GPSLatitude', 'GPSLongitude'] and isinstance(gps_value, tuple) and len(gps_value) == 3:
                                    try:
                                        # Convert DMS (Degrees, Minutes, Seconds) to decimal
                                        degrees = float(gps_value[0])
                                        minutes = float(gps_value[1])
                                        seconds = float(gps_value[2])
                                        decimal = degrees + minutes/60 + seconds/3600
                                        
                                        gps_data[gps_tag] = f"{int(degrees)}° {int(minutes)}' {seconds:.2f}\""
                                        gps_data[f'{gps_tag}_Decimal'] = round(decimal, 6)
                                    except:
                                        gps_data[gps_tag] = str(gps_value)
                                elif gps_tag == 'GPSTimeStamp' and isinstance(gps_value, tuple):
                                    try:
                                        gps_data[gps_tag] = f"{int(gps_value[0]):02d}:{int(gps_value[1]):02d}:{int(gps_value[2]):02d}"
                                    except:
                                        gps_data[gps_tag] = str(gps_value)
                                elif isinstance(gps_value, bytes):
                                    try:
                                        gps_data[gps_tag] = gps_value.decode('utf-8', errors='ignore')
                                    except:
                                        gps_data[gps_tag] = str(gps_value)
                                elif hasattr(gps_value, 'numerator') and hasattr(gps_value, 'denominator'):
                                    try:
                                        gps_data[gps_tag] = float(gps_value)
                                    except:
                                        gps_data[gps_tag] = str(gps_value)
                                else:
                                    gps_data[gps_tag] = gps_value
                            
                            if gps_data:
                                exif['GPSInfo'] = gps_data
                    except (KeyError, AttributeError):
                        pass
                    
                    if exif:
                        metadata['EXIF'] = exif
                
                # PNG specific metadata
                if img.format == 'PNG':
                    png_info = {}
                    for key, value in img.info.items():
                        png_info[key] = value
                    if png_info:
                        metadata['PNGInfo'] = png_info
                
                # Check for thumbnail data
                if hasattr(img, 'thumbnail') or 'thumbnail' in img.info:
                    metadata['HasThumbnail'] = True
                
                return metadata
                
        except Exception as e:
            return {'error': f'Failed to extract image metadata: {str(e)}'}
    
    def _extract_pdf_metadata(self) -> Dict[str, Any]:
        """Extract metadata from PDF files"""
        if not PYPDF2_AVAILABLE:
            return {'error': 'PyPDF2 library not available'}
        
        try:
            with open(self.filepath, 'rb') as f:
                pdf = PdfReader(f)
                
                metadata = {
                    'PageCount': len(pdf.pages),
                    'IsEncrypted': pdf.is_encrypted,
                }
                
                # Extract document info
                if pdf.metadata:
                    for key, value in pdf.metadata.items():
                        clean_key = key.replace('/', '') if key.startswith('/') else key
                        metadata[clean_key] = value
                
                # Extract PDF version
                try:
                    metadata['PDFVersion'] = pdf.pdf_header
                except:
                    pass
                
                # Extract first page dimensions
                if len(pdf.pages) > 0:
                    first_page = pdf.pages[0]
                    if hasattr(first_page, 'mediabox'):
                        mediabox = first_page.mediabox
                        metadata['PageWidth'] = float(mediabox.width)
                        metadata['PageHeight'] = float(mediabox.height)
                        metadata['PageSize'] = f"{float(mediabox.width)} x {float(mediabox.height)}"
                
                return metadata
                
        except Exception as e:
            return {'error': f'Failed to extract PDF metadata: {str(e)}'}
    
    def _extract_audio_metadata(self) -> Dict[str, Any]:
        """Extract metadata from audio files"""
        if not MUTAGEN_AVAILABLE:
            return {'error': 'Mutagen library not available'}
        
        try:
            audio = MutagenFile(self.filepath)
            
            if audio is None:
                return {'error': 'Unsupported audio format'}
            
            metadata = {
                'Duration': round(audio.info.length, 2) if hasattr(audio.info, 'length') else None,
                'DurationHuman': self._format_duration(audio.info.length) if hasattr(audio.info, 'length') else None,
                'Bitrate': audio.info.bitrate if hasattr(audio.info, 'bitrate') else None,
                'BitrateKbps': round(audio.info.bitrate / 1000, 2) if hasattr(audio.info, 'bitrate') else None,
                'SampleRate': audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else None,
                'Channels': audio.info.channels if hasattr(audio.info, 'channels') else None,
                'BitsPerSample': getattr(audio.info, 'bits_per_sample', None),
            }
            
            # Extract tags
            if audio.tags:
                tags = {}
                for key, value in audio.tags.items():
                    if isinstance(value, list):
                        tags[key] = ', '.join(str(v) for v in value)
                    else:
                        tags[key] = str(value)
                metadata['Tags'] = tags
            
            return metadata
            
        except Exception as e:
            return {'error': f'Failed to extract audio metadata: {str(e)}'}
    
    def _extract_video_metadata(self) -> Dict[str, Any]:
        """Extract metadata from video files"""
        if not FFMPEG_AVAILABLE:
            return {'error': 'ffmpeg-python library not available'}
        
        try:
            probe = ffmpeg.probe(str(self.filepath))
            
            metadata = {
                'Format': probe.get('format', {}).get('format_long_name'),
                'FormatName': probe.get('format', {}).get('format_name'),
                'Duration': float(probe.get('format', {}).get('duration', 0)),
                'DurationHuman': self._format_duration(float(probe.get('format', {}).get('duration', 0))),
                'Bitrate': int(probe.get('format', {}).get('bit_rate', 0)),
                'BitrateKbps': round(int(probe.get('format', {}).get('bit_rate', 0)) / 1000, 2),
                'Size': int(probe.get('format', {}).get('size', 0)),
                'StreamCount': probe.get('format', {}).get('nb_streams'),
            }
            
            # Extract video stream info
            video_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
            if video_streams:
                video = video_streams[0]
                metadata['VideoCodec'] = video.get('codec_name')
                metadata['VideoCodecLong'] = video.get('codec_long_name')
                metadata['VideoWidth'] = video.get('width')
                metadata['VideoHeight'] = video.get('height')
                metadata['VideoResolution'] = f"{video.get('width')}x{video.get('height')}"
                metadata['FrameRate'] = video.get('r_frame_rate')
                metadata['AvgFrameRate'] = video.get('avg_frame_rate')
                metadata['AspectRatio'] = video.get('display_aspect_ratio')
                metadata['PixelFormat'] = video.get('pix_fmt')
                metadata['ColorSpace'] = video.get('color_space')
            
            # Extract audio stream info
            audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
            if audio_streams:
                audio = audio_streams[0]
                metadata['AudioCodec'] = audio.get('codec_name')
                metadata['AudioCodecLong'] = audio.get('codec_long_name')
                metadata['AudioSampleRate'] = audio.get('sample_rate')
                metadata['AudioChannels'] = audio.get('channels')
                metadata['AudioChannelLayout'] = audio.get('channel_layout')
            
            # Extract format tags
            if 'tags' in probe.get('format', {}):
                metadata['Tags'] = probe['format']['tags']
            
            return metadata
            
        except Exception as e:
            return {'error': f'Failed to extract video metadata: {str(e)}'}
    
    def _redact_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact sensitive information from metadata"""
        
        def redact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            redacted = {}
            for key, value in d.items():
                key_lower = key.lower()
                
                # Check if key is sensitive
                if any(sensitive in key_lower for sensitive in self.SENSITIVE_FIELDS):
                    redacted[key] = '[REDACTED]'
                elif isinstance(value, dict):
                    redacted[key] = redact_dict(value)
                elif isinstance(value, list):
                    redacted[key] = [redact_dict(item) if isinstance(item, dict) else item for item in value]
                else:
                    redacted[key] = value
            
            return redacted
        
        return redact_dict(metadata)
    
    def _get_mime_type(self) -> str:
        """Get MIME type of the file"""
        mime_type, _ = mimetypes.guess_type(str(self.filepath))
        return mime_type or 'application/octet-stream'
    
    def _calculate_hash(self, algorithm: str = 'md5') -> str:
        """Calculate file hash"""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(self.filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception:
            return None
    
    def _get_permission_string(self, mode: int) -> str:
        """Convert file permissions to readable string"""
        perms = ''
        perms += 'r' if mode & 0o400 else '-'
        perms += 'w' if mode & 0o200 else '-'
        perms += 'x' if mode & 0o100 else '-'
        perms += 'r' if mode & 0o040 else '-'
        perms += 'w' if mode & 0o020 else '-'
        perms += 'x' if mode & 0o010 else '-'
        perms += 'r' if mode & 0o004 else '-'
        perms += 'w' if mode & 0o002 else '-'
        perms += 'x' if mode & 0o001 else '-'
        return perms
    
    @staticmethod
    def _human_readable_size(size: int) -> str:
        """Convert bytes to human readable format"""
        for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Convert seconds to human readable duration"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


class MetadataFormatter:
    """Format metadata for different output types"""
    
    @staticmethod
    def to_json(metadata: Dict[str, Any], indent: int = 2) -> str:
        """Format metadata as JSON"""
        return json.dumps(metadata, indent=indent, default=str)
    
    @staticmethod
    def to_human_readable(metadata: Dict[str, Any]) -> str:
        """Format metadata in human-readable table format"""
        output = []
        output.append("=" * 80)
        output.append("METADATA EXTRACTION REPORT")
        output.append("=" * 80)
        output.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        output.append("")
        
        def print_section(data: Dict[str, Any], section_name: str = None, indent: int = 0):
            """Recursively print metadata sections"""
            if section_name:
                output.append("  " * indent + f"[{section_name.upper()}]")
                output.append("  " * indent + "-" * 60)
            
            for key, value in data.items():
                if isinstance(value, dict):
                    output.append("")
                    print_section(value, key, indent + 1)
                elif isinstance(value, list):
                    if len(value) > 0:
                        output.append("  " * indent + f"  {key:40} : ")
                        for item in value:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    output.append("  " * indent + f"    - {k}: {v}")
                            else:
                                output.append("  " * indent + f"    - {item}")
                    else:
                        output.append("  " * indent + f"  {key:40} : {value}")
                else:
                    output.append("  " * indent + f"  {key:40} : {value}")
            
            output.append("")
        
        for section, data in metadata.items():
            if isinstance(data, dict):
                print_section(data, section)
            else:
                output.append(f"{section:40} : {data}")
        
        output.append("=" * 80)
        output.append("END OF REPORT")
        output.append("=" * 80)
        return "\n".join(output)


def extract_metadata_from_file(filepath: str, output_format: str = 'human', 
                               redact_sensitive: bool = False) -> Union[str, Dict]:
    """
    Extract metadata from a file
    
    Args:
        filepath: Path to the file
        output_format: 'json' or 'human' (default: 'human')
        redact_sensitive: Redact sensitive information like GPS, serial numbers (default: False)
    
    Returns:
        Formatted metadata string or dictionary
    """
    try:
        extractor = MetadataExtractor(filepath, redact_sensitive=redact_sensitive)
        metadata = extractor.extract_all()
        
        if output_format.lower() == 'json':
            return MetadataFormatter.to_json(metadata)
        else:
            return MetadataFormatter.to_human_readable(metadata)
            
    except Exception as e:
        return f"Error extracting metadata: {str(e)}"


if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print(" " * 20 + "METADATA EXTRACTOR")
    print("=" * 80)
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python metadata_extractor.py <file_path> [output_format] [--redact]")
        print()
        print("Arguments:")
        print("  file_path       : Path to the file to analyze (required)")
        print("  output_format   : 'json' or 'human' (default: human)")
        print("  --redact        : Redact sensitive information (GPS, serial numbers, etc.)")
        print()
        print("Examples:")
        print("  python metadata_extractor.py photo.jpg")
        print("  python metadata_extractor.py document.pdf json")
        print("  python metadata_extractor.py image.png human --redact")
        print("  python metadata_extractor.py video.mp4 json --redact")
        print()
        print("Supported File Types:")
        print("  • Images: PNG, JPG, TIFF, BMP, GIF")
        print("  • Documents: PDF")
        print("  • Audio: MP3, FLAC, WAV, OGG")
        print("  • Video: MP4, AVI, MKV, WebM")
        print()
        print("Note: This tool only reads metadata. It does not execute or modify files.")
        sys.exit(1)
    
    filepath = sys.argv[1]
    output_format = 'human'
    redact = False
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg.lower() in ['json', 'human']:
            output_format = arg.lower()
        elif arg.lower() == '--redact':
            redact = True
    
    if redact:
        print("ℹ️  REDACTION MODE ENABLED - Sensitive data will be redacted")
        print()
    
    result = extract_metadata_from_file(filepath, output_format, redact_sensitive=redact)
    print(result)
