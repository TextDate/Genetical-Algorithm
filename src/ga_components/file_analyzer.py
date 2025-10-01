"""
File Analysis and Data Type Classification Component

Provides automatic file type detection, data domain classification, and 
characteristics analysis for multi-domain compression optimization.
"""

import os
import math
import mimetypes
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import struct

from ga_logging import get_logger


class DataType(Enum):
    """Primary data type categories."""
    TEXT = "text"
    BINARY = "binary" 
    SCIENTIFIC = "scientific"
    MULTIMEDIA = "multimedia"
    CODE = "code"
    COMPRESSED = "compressed"
    DATABASE = "database"
    UNKNOWN = "unknown"


class DataDomain(Enum):
    """Specific data domain classifications."""
    NATURAL_LANGUAGE = "natural_language"
    SOURCE_CODE = "source_code"
    CONFIGURATION = "configuration"
    MARKUP = "markup"
    CHEMICAL_DATA = "chemical_data"
    GENOMIC_DATA = "genomic_data"
    NUMERICAL_DATA = "numerical_data"
    IMAGE_DATA = "image_data"
    AUDIO_DATA = "audio_data"
    VIDEO_DATA = "video_data"
    EXECUTABLE = "executable"
    LIBRARY = "library"
    ARCHIVE = "archive"
    DATABASE_FILE = "database_file"
    LOG_DATA = "log_data"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class FileCharacteristics:
    """File characteristics and metadata."""
    file_path: str
    file_size: int
    data_type: DataType
    data_domain: DataDomain
    
    # Compression-relevant metrics
    entropy: float
    predicted_compressibility: str  # "high", "medium", "low"
    
    # File structure analysis
    text_ratio: float  # 0.0 - 1.0
    binary_ratio: float  # 0.0 - 1.0
    repetition_factor: float  # 0.0 - 1.0 (higher = more repetitive)
    
    # Recommended compressors (ordered by predicted performance)
    recommended_compressors: List[str]
    
    # Additional metadata
    mime_type: Optional[str] = None
    file_extension: Optional[str] = None
    detected_encoding: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'file_path': self.file_path,
            'file_size': self.file_size,
            'data_type': self.data_type.value,
            'data_domain': self.data_domain.value,
            'entropy': self.entropy,
            'predicted_compressibility': self.predicted_compressibility,
            'text_ratio': self.text_ratio,
            'binary_ratio': self.binary_ratio,
            'repetition_factor': self.repetition_factor,
            'recommended_compressors': self.recommended_compressors,
            'mime_type': self.mime_type,
            'file_extension': self.file_extension,
            'detected_encoding': self.detected_encoding
        }


class FileAnalyzer:
    """
    Comprehensive file analysis for multi-domain compression optimization.
    
    Analyzes files to determine data type, domain, characteristics, and
    provides compressor recommendations based on data properties.
    """
    
    # Magic number patterns for file type detection
    MAGIC_NUMBERS = {
        # Archives and compressed
        b'\x1f\x8b': ('gzip', DataType.COMPRESSED),
        b'PK\x03\x04': ('zip', DataType.COMPRESSED),
        b'PK\x05\x06': ('zip', DataType.COMPRESSED),
        b'PK\x07\x08': ('zip', DataType.COMPRESSED),
        b'\x42\x5a\x68': ('bzip2', DataType.COMPRESSED),
        b'\x7f\x45\x4c\x46': ('elf', DataType.BINARY),
        
        # Images
        b'\xff\xd8\xff': ('jpeg', DataType.MULTIMEDIA),
        b'\x89PNG\r\n\x1a\n': ('png', DataType.MULTIMEDIA),
        b'GIF87a': ('gif', DataType.MULTIMEDIA),
        b'GIF89a': ('gif', DataType.MULTIMEDIA),
        
        # Audio/Video
        b'ID3': ('mp3', DataType.MULTIMEDIA),
        b'\xff\xfb': ('mp3', DataType.MULTIMEDIA),
        b'RIFF': ('wav/avi', DataType.MULTIMEDIA),
        
        # Documents
        b'%PDF': ('pdf', DataType.TEXT),
        b'\xd0\xcf\x11\xe0': ('ms_office', DataType.TEXT),
        
        # Database
        b'SQLite format 3\x00': ('sqlite', DataType.DATABASE),
    }
    
    # Text file extensions
    TEXT_EXTENSIONS = {
        '.txt', '.log', '.csv', '.json', '.xml', '.html', '.htm', '.md', '.rst',
        '.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb',
        '.go', '.rs', '.scala', '.swift', '.kt', '.dart', '.ts', '.jsx', '.tsx',
        '.vue', '.svelte', '.css', '.scss', '.sass', '.less', '.sql', '.yml', '.yaml',
        '.toml', '.ini', '.cfg', '.conf', '.properties', '.env'
    }
    
    # Scientific data extensions  
    SCIENTIFIC_EXTENSIONS = {
        '.sdf', '.mol', '.pdb', '.xyz', '.cif', '.mmcif', '.fasta', '.fa', '.fastq',
        '.sam', '.bam', '.vcf', '.gff', '.gtf', '.bed', '.wig', '.bedgraph',
        '.hdf5', '.h5', '.nc', '.cdf', '.mat', '.npy', '.npz'
    }
    
    def __init__(self):
        self.logger = get_logger("FileAnalyzer")
        
    def analyze_file(self, file_path: str) -> FileCharacteristics:
        """
        Comprehensive analysis of a file to determine its characteristics.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileCharacteristics object with analysis results
        """
        self.logger.debug(f"Analyzing file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return self._create_empty_file_characteristics(file_path)
        
        # Get basic file info
        file_extension = os.path.splitext(file_path)[1].lower()
        mime_type = mimetypes.guess_type(file_path)[0]
        
        # Read file sample for analysis (limit to avoid memory issues)
        sample_size = min(file_size, 1024 * 1024)  # 1MB max sample
        with open(file_path, 'rb') as f:
            sample_data = f.read(sample_size)
        
        # Detect file type via magic numbers
        detected_type, detected_domain = self._detect_type_from_magic(sample_data)
        
        # If magic number detection fails, use extension-based detection
        if detected_type == DataType.UNKNOWN:
            detected_type, detected_domain = self._detect_type_from_extension(file_extension)
        
        # Analyze content characteristics
        entropy = self._calculate_entropy(sample_data)
        text_ratio = self._calculate_text_ratio(sample_data)
        binary_ratio = 1.0 - text_ratio
        repetition_factor = self._calculate_repetition_factor(sample_data)
        
        # Predict compressibility
        predicted_compressibility = self._predict_compressibility(
            entropy, text_ratio, repetition_factor, detected_type
        )
        
        # Generate compressor recommendations
        recommended_compressors = self._recommend_compressors(
            detected_type, detected_domain, entropy, text_ratio, 
            repetition_factor, file_size
        )
        
        # Detect text encoding if applicable
        detected_encoding = None
        if text_ratio > 0.7:
            detected_encoding = self._detect_encoding(sample_data)
        
        characteristics = FileCharacteristics(
            file_path=file_path,
            file_size=file_size,
            data_type=detected_type,
            data_domain=detected_domain,
            entropy=entropy,
            predicted_compressibility=predicted_compressibility,
            text_ratio=text_ratio,
            binary_ratio=binary_ratio,
            repetition_factor=repetition_factor,
            recommended_compressors=recommended_compressors,
            mime_type=mime_type,
            file_extension=file_extension,
            detected_encoding=detected_encoding
        )
        
        self.logger.debug(f"Analysis complete: {detected_type.value}/{detected_domain.value}, "
                         f"compressibility: {predicted_compressibility}")
        
        return characteristics
    
    def _detect_type_from_magic(self, data: bytes) -> Tuple[DataType, DataDomain]:
        """Detect file type using magic number patterns."""
        for magic, (file_type, data_type) in self.MAGIC_NUMBERS.items():
            if data.startswith(magic):
                domain = self._map_file_type_to_domain(file_type)
                return data_type, domain
        
        return DataType.UNKNOWN, DataDomain.UNKNOWN
    
    def _detect_type_from_extension(self, extension: str) -> Tuple[DataType, DataDomain]:
        """Detect file type using file extension."""
        if extension in self.TEXT_EXTENSIONS:
            if extension in {'.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', 
                           '.php', '.rb', '.go', '.rs', '.scala', '.swift', '.kt', '.dart',
                           '.ts', '.jsx', '.tsx', '.vue', '.svelte'}:
                return DataType.CODE, DataDomain.SOURCE_CODE
            elif extension in {'.html', '.htm', '.xml', '.md', '.rst'}:
                return DataType.TEXT, DataDomain.MARKUP
            elif extension in {'.yml', '.yaml', '.toml', '.ini', '.cfg', '.conf', '.properties', '.env'}:
                return DataType.TEXT, DataDomain.CONFIGURATION
            elif extension in {'.log'}:
                return DataType.TEXT, DataDomain.LOG_DATA
            elif extension in {'.sql'}:
                return DataType.CODE, DataDomain.SOURCE_CODE
            else:
                return DataType.TEXT, DataDomain.NATURAL_LANGUAGE
        
        elif extension in self.SCIENTIFIC_EXTENSIONS:
            if extension in {'.sdf', '.mol', '.pdb', '.xyz', '.cif', '.mmcif'}:
                return DataType.SCIENTIFIC, DataDomain.CHEMICAL_DATA
            elif extension in {'.fasta', '.fa', '.fastq', '.sam', '.bam', '.vcf', '.gff', '.gtf'}:
                return DataType.SCIENTIFIC, DataDomain.GENOMIC_DATA
            else:
                return DataType.SCIENTIFIC, DataDomain.NUMERICAL_DATA
        
        elif extension in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}:
            return DataType.MULTIMEDIA, DataDomain.IMAGE_DATA
        
        elif extension in {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}:
            return DataType.MULTIMEDIA, DataDomain.AUDIO_DATA
        
        elif extension in {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}:
            return DataType.MULTIMEDIA, DataDomain.VIDEO_DATA
        
        elif extension in {'.exe', '.dll', '.so', '.dylib', '.bin'}:
            return DataType.BINARY, DataDomain.EXECUTABLE
        
        elif extension in {'.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar'}:
            return DataType.COMPRESSED, DataDomain.ARCHIVE
        
        elif extension in {'.db', '.sqlite', '.sqlite3', '.mdb', '.accdb'}:
            return DataType.DATABASE, DataDomain.DATABASE_FILE
        
        return DataType.UNKNOWN, DataDomain.UNKNOWN
    
    def _map_file_type_to_domain(self, file_type: str) -> DataDomain:
        """Map detected file type to data domain."""
        mapping = {
            'gzip': DataDomain.ARCHIVE,
            'zip': DataDomain.ARCHIVE, 
            'bzip2': DataDomain.ARCHIVE,
            'elf': DataDomain.EXECUTABLE,
            'jpeg': DataDomain.IMAGE_DATA,
            'png': DataDomain.IMAGE_DATA,
            'gif': DataDomain.IMAGE_DATA,
            'mp3': DataDomain.AUDIO_DATA,
            'wav': DataDomain.AUDIO_DATA,
            'avi': DataDomain.VIDEO_DATA,
            'pdf': DataDomain.NATURAL_LANGUAGE,
            'ms_office': DataDomain.NATURAL_LANGUAGE,
            'sqlite': DataDomain.DATABASE_FILE,
        }
        return mapping.get(file_type, DataDomain.UNKNOWN)
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_text_ratio(self, data: bytes) -> float:
        """Calculate ratio of printable text characters."""
        if not data:
            return 0.0
        
        printable_count = 0
        for byte in data:
            if 32 <= byte <= 126 or byte in {9, 10, 13}:  # Printable ASCII + tab/newline/CR
                printable_count += 1
        
        return printable_count / len(data)
    
    def _calculate_repetition_factor(self, data: bytes) -> float:
        """Calculate repetition factor indicating redundancy."""
        if len(data) < 2:
            return 0.0
        
        # Simple approach: count repeated byte sequences
        repeated_bytes = 0
        for i in range(len(data) - 1):
            if data[i] == data[i + 1]:
                repeated_bytes += 1
        
        return repeated_bytes / (len(data) - 1)
    
    def _predict_compressibility(self, entropy: float, text_ratio: float, 
                               repetition_factor: float, data_type: DataType) -> str:
        """Predict compressibility based on file characteristics."""
        # Already compressed files
        if data_type == DataType.COMPRESSED:
            return "low"
        
        # High entropy indicates randomness (low compressibility)
        if entropy > 7.5:
            return "low"
        elif entropy < 4.0:
            return "high"
        
        # Text files with high repetition compress well
        if text_ratio > 0.8 and repetition_factor > 0.3:
            return "high"
        
        # Scientific data often compresses well due to patterns
        if data_type == DataType.SCIENTIFIC:
            return "medium" if entropy > 6.0 else "high"
        
        # Code files typically compress well
        if data_type == DataType.CODE:
            return "high"
        
        # Multimedia files often already optimized
        if data_type == DataType.MULTIMEDIA:
            return "low"
        
        # Default classification based on entropy
        if entropy < 5.5:
            return "high"
        elif entropy < 6.5:
            return "medium" 
        else:
            return "low"
    
    def _recommend_compressors(self, data_type: DataType, data_domain: DataDomain,
                              entropy: float, text_ratio: float, repetition_factor: float,
                              file_size: int) -> List[str]:
        """Generate ranked compressor recommendations."""
        recommendations = []
        
        # Already compressed - minimal benefit from further compression
        if data_type == DataType.COMPRESSED:
            return ['zstd']  # Fast decompression if needed
        
        # Text-based data (high text ratio)
        if text_ratio > 0.8:
            if data_domain == DataDomain.SOURCE_CODE:
                recommendations = ['lzma', 'brotli', 'zstd']  # Code compresses very well
            elif data_domain == DataDomain.NATURAL_LANGUAGE:
                recommendations = ['brotli', 'lzma', 'zstd']  # Brotli excellent for text
            elif data_domain == DataDomain.LOG_DATA:
                recommendations = ['lzma', 'zstd', 'brotli']  # Logs often repetitive
            else:
                recommendations = ['brotli', 'lzma', 'zstd']
        
        # Scientific data
        elif data_type == DataType.SCIENTIFIC:
            if data_domain == DataDomain.CHEMICAL_DATA:
                recommendations = ['lzma', 'brotli', 'zstd']  # Structured data
            elif data_domain == DataDomain.GENOMIC_DATA:
                recommendations = ['lzma', 'zstd', 'brotli']  # Highly repetitive
            else:
                recommendations = ['lzma', 'zstd', 'brotli']
        
        # Binary data
        elif data_type == DataType.BINARY:
            if entropy < 6.0:  # Low entropy binary
                recommendations = ['lzma', 'zstd', 'brotli']
            else:  # High entropy binary
                recommendations = ['zstd', 'brotli', 'lzma']  # Speed over ratio
        
        # Multimedia - typically pre-compressed
        elif data_type == DataType.MULTIMEDIA:
            recommendations = ['zstd']  # Focus on speed
        
        # Database files
        elif data_type == DataType.DATABASE:
            recommendations = ['lzma', 'zstd', 'brotli']  # Often structured/repetitive
        
        # Default recommendations based on characteristics
        else:
            if entropy < 5.0:  # Very low entropy
                recommendations = ['lzma', 'brotli', 'zstd']
            elif entropy < 6.5:  # Medium entropy
                recommendations = ['brotli', 'lzma', 'zstd']
            else:  # High entropy
                recommendations = ['zstd', 'brotli', 'lzma']
        
        # Adjust for file size
        if file_size > 100 * 1024 * 1024:  # Large files (>100MB)
            # Prioritize speed for large files
            if 'zstd' in recommendations:
                recommendations.remove('zstd')
                recommendations.insert(0, 'zstd')
        
        # Ensure we have recommendations
        if not recommendations:
            recommendations = ['zstd', 'brotli', 'lzma']
        
        return recommendations
    
    def _detect_encoding(self, data: bytes) -> Optional[str]:
        """Detect text encoding."""
        try:
            # Try UTF-8 first
            data.decode('utf-8')
            return 'utf-8'
        except UnicodeDecodeError:
            pass
        
        try:
            # Try Latin-1
            data.decode('latin-1')
            return 'latin-1'
        except UnicodeDecodeError:
            pass
        
        # Try ASCII
        try:
            data.decode('ascii')
            return 'ascii'
        except UnicodeDecodeError:
            return None
    
    def _create_empty_file_characteristics(self, file_path: str) -> FileCharacteristics:
        """Create characteristics for empty file."""
        return FileCharacteristics(
            file_path=file_path,
            file_size=0,
            data_type=DataType.UNKNOWN,
            data_domain=DataDomain.UNKNOWN,
            entropy=0.0,
            predicted_compressibility="low",
            text_ratio=0.0,
            binary_ratio=0.0,
            repetition_factor=0.0,
            recommended_compressors=["zstd"],
            file_extension=os.path.splitext(file_path)[1].lower(),
        )