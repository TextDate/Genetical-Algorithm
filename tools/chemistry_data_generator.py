#!/usr/bin/env python3
"""
Chemistry Data Generator - Create standardized size datasets from large .sdf.gz files

This tool extracts portions of the chemistry dataset to create files of specific sizes
(1MB, 10MB, 50MB, 100MB, 500MB, 1GB) for systematic compressor benchmarking.

Usage:
    python tools/chemistry_data_generator.py --source_dir ../Chem_Dataset/Chem_dataset/ --output_dir data/chemistry_samples/
"""

import os
import sys
import gzip
import argparse
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChemistryDataGenerator:
    """Generate standardized size chemistry datasets from large compressed files."""
    
    # Target sizes in bytes
    TARGET_SIZES = [
        (1024 * 1024, "1mb"),           # 1 MB
        (5 * 1024 * 1024, "5mb"),       # 5 MB
        (10 * 1024 * 1024, "10mb"),     # 10 MB  
        (50 * 1024 * 1024, "50mb"),     # 50 MB
        (100 * 1024 * 1024, "100mb"),   # 100 MB
        (500 * 1024 * 1024, "500mb"),   # 500 MB
        (1024 * 1024 * 1024, "1gb"),    # 1 GB
    ]
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Create output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)
        for _, size_name in self.TARGET_SIZES:
            size_dir = os.path.join(self.output_dir, size_name)
            os.makedirs(size_dir, exist_ok=True)
    
    def find_source_files(self) -> List[str]:
        """Find all .sdf.gz files in source directory."""
        files = []
        if not os.path.exists(self.source_dir):
            logger.error(f"Source directory not found: {self.source_dir}")
            return files
            
        for file in os.listdir(self.source_dir):
            if file.endswith('.sdf.gz'):
                files.append(os.path.join(self.source_dir, file))
        
        files.sort()  # Process in order
        logger.info(f"Found {len(files)} source files")
        return files
    
    def extract_size_sample(self, source_file: str, target_size: int, output_file: str) -> bool:
        """Extract a portion of source file to match target size."""
        try:
            logger.info(f"Creating {output_file} (target: {target_size / (1024*1024):.1f}MB)")
            
            bytes_written = 0
            compounds_written = 0
            
            with gzip.open(source_file, 'rt', encoding='utf-8', errors='ignore') as infile:
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    
                    current_compound = []
                    in_compound = False
                    
                    for line in infile:
                        current_compound.append(line)
                        
                        # Check if we're at the end of a compound
                        if line.strip() == "$$$$":
                            # Write complete compound
                            compound_text = ''.join(current_compound)
                            outfile.write(compound_text)
                            
                            bytes_written += len(compound_text.encode('utf-8'))
                            compounds_written += 1
                            
                            # Check if we've reached target size
                            if bytes_written >= target_size:
                                break
                            
                            current_compound = []
                            
                            # Log progress every 1000 compounds
                            if compounds_written % 1000 == 0:
                                logger.info(f"  Progress: {compounds_written} compounds, {bytes_written / (1024*1024):.1f}MB")
            
            actual_size = os.path.getsize(output_file)
            logger.info(f"  Completed: {compounds_written} compounds, {actual_size / (1024*1024):.1f}MB actual size")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {source_file}: {e}")
            return False
    
    def generate_samples(self, max_samples_per_size: int = 3):
        """Generate sample files for each target size."""
        source_files = self.find_source_files()
        if not source_files:
            logger.error("No source files found!")
            return
        
        logger.info(f"Generating samples with max {max_samples_per_size} per size")
        
        for target_size, size_name in self.TARGET_SIZES:
            logger.info(f"\nGenerating {size_name.upper()} samples:")
            
            samples_created = 0
            source_file_idx = 0
            
            while samples_created < max_samples_per_size and source_file_idx < len(source_files):
                source_file = source_files[source_file_idx]
                
                # Check if source file is large enough
                if os.path.getsize(source_file) < target_size:
                    logger.warning(f"Source file {os.path.basename(source_file)} too small for {size_name}")
                    source_file_idx += 1
                    continue
                
                # Generate output filename
                base_name = os.path.basename(source_file).replace('.sdf.gz', '')
                output_file = os.path.join(
                    self.output_dir, 
                    size_name, 
                    f"{base_name}_{size_name}.sdf"
                )
                
                # Skip if already exists
                if os.path.exists(output_file):
                    logger.info(f"  Skipping existing file: {os.path.basename(output_file)}")
                    samples_created += 1
                    source_file_idx += 1
                    continue
                
                # Extract sample
                if self.extract_size_sample(source_file, target_size, output_file):
                    samples_created += 1
                    logger.info(f"  Created: {os.path.basename(output_file)}")
                else:
                    logger.error(f"  Failed to create: {os.path.basename(output_file)}")
                
                source_file_idx += 1
            
            logger.info(f"Completed {size_name}: {samples_created} samples created")
    
    def create_sample_readme(self):
        """Create README for the generated samples."""
        readme_path = os.path.join(self.output_dir, "README.md")
        
        readme_content = f"""# Chemistry Dataset Samples

Generated from: `{os.path.abspath(self.source_dir)}`

## Sample Sizes Available

| Size | Directory | Purpose |
|------|-----------|---------|
| 1MB  | `1mb/`    | Quick testing, development |
| 10MB | `10mb/`   | Small-scale benchmarking |  
| 50MB | `50mb/`   | Medium-scale testing |
| 100MB| `100mb/`  | Large-scale benchmarking |
| 500MB| `500mb/`  | High-performance testing |
| 1GB  | `1gb/`    | Maximum scale testing |

## File Format
- **Format**: SDF (Structure Data Format) - uncompressed
- **Content**: Chemical compound structures and properties
- **Encoding**: UTF-8 text format
- **Structure**: Multiple compounds per file, separated by `$$$$`

## Usage Examples

```bash
# Test with 1MB sample
python main.py --compressor zstd \\
  --param_file config/params.json \\
  --input data/chemistry_samples/1mb/Compound_000000001_000500000_1mb.sdf

# Benchmark with 100MB sample  
python main.py --compressor lzma \\
  --param_file config/params.json \\
  --input data/chemistry_samples/100mb/Compound_001000001_001500000_100mb.sdf

# Batch test multiple sizes
for size in 1mb 10mb 50mb 100mb; do
  python main.py --compressor brotli \\
    --param_file config/params.json \\
    --input data/chemistry_samples/$size/*.sdf \\
    --output_dir results_chemistry_$size
done
```

## Integration with Activity Plan

### Phase 1 (Foundation): Use 1MB and 10MB samples
- Quick development and testing
- Initial compressor performance baseline
- Algorithm validation

### Phase 2 (Analysis): Use 50MB and 100MB samples  
- Multi-domain performance analysis
- Statistical significance testing
- Pattern identification

### Phase 3+ (Scale): Use 500MB and 1GB samples
- Large-scale validation
- Performance scaling analysis
- Production-ready benchmarking

Generated on: {os.popen('date').read().strip()}
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created README: {readme_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate standardized chemistry dataset samples')
    parser.add_argument('--source_dir', '-s', 
                       default='../Chem_Dataset/Chem_dataset/', 
                       help='Source directory containing .sdf.gz files')
    parser.add_argument('--output_dir', '-o', 
                       default='data/chemistry_samples/', 
                       help='Output directory for generated samples')
    parser.add_argument('--max_samples', '-m', 
                       type=int, default=3, 
                       help='Maximum samples to generate per size')
    
    args = parser.parse_args()
    
    generator = ChemistryDataGenerator(args.source_dir, args.output_dir)
    
    logger.info("Chemistry Data Generator Starting")
    logger.info(f"Source: {os.path.abspath(args.source_dir)}")
    logger.info(f"Output: {os.path.abspath(args.output_dir)}")
    
    # Generate samples
    generator.generate_samples(args.max_samples)
    
    # Create documentation
    generator.create_sample_readme()
    
    logger.info("Generation complete!")


if __name__ == "__main__":
    main()