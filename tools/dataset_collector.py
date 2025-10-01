"""
Comprehensive Multi-Domain Dataset Collection Tool

Creates and manages diverse datasets for multi-domain compression testing.
Generates synthetic data across different domains and organizes real-world datasets.
"""

import os
import json
import random
import string
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# For dataset generation
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


@dataclass 
class DatasetSpec:
    """Specification for dataset generation."""
    domain: str
    data_type: str
    file_count: int
    size_range: Tuple[int, int]  # (min_bytes, max_bytes)
    characteristics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain': self.domain,
            'data_type': self.data_type,
            'file_count': self.file_count,
            'size_range': self.size_range,
            'characteristics': self.characteristics
        }


class MultiDomainDatasetCollector:
    """
    Comprehensive dataset collection and generation system.
    
    Creates diverse datasets across multiple domains for thorough
    compression algorithm testing and evaluation.
    """
    
    def __init__(self, output_base_dir: str = "datasets"):
        self.output_base_dir = Path(output_base_dir)
        self.logger = self._setup_logging()
        
        # Dataset specifications for different domains
        self.dataset_specs = self._define_dataset_specifications()
        
    def create_comprehensive_dataset_collection(self) -> Dict[str, Any]:
        """
        Create a comprehensive collection of multi-domain datasets.
        
        Returns:
            Dictionary with dataset creation summary and metadata
        """
        self.logger.info("Starting comprehensive dataset collection creation")
        
        # Create base directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        created_datasets = {}
        
        # Generate datasets for each specification
        for domain, specs in self.dataset_specs.items():
            self.logger.info(f"Creating datasets for domain: {domain}")
            
            domain_dir = self.output_base_dir / domain
            domain_dir.mkdir(exist_ok=True)
            
            domain_results = []
            
            for spec in specs:
                try:
                    result = self._generate_dataset(spec, domain_dir)
                    domain_results.append(result)
                    self.logger.info(f"Created {spec.data_type} dataset: {result['file_count']} files")
                except Exception as e:
                    self.logger.error(f"Failed to create {spec.data_type} dataset: {e}")
            
            created_datasets[domain] = domain_results
        
        # Create dataset index and metadata
        dataset_metadata = self._create_dataset_metadata(created_datasets)
        
        # Save metadata
        metadata_file = self.output_base_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2, default=str)
        
        # Create README
        self._create_dataset_readme(dataset_metadata)
        
        self.logger.info(f"Dataset collection completed. Output: {self.output_base_dir}")
        return dataset_metadata
    
    def augment_existing_datasets(self, dataset_dirs: List[str]) -> Dict[str, Any]:
        """
        Augment existing datasets with metadata and organization.
        
        Args:
            dataset_dirs: List of existing dataset directories to process
            
        Returns:
            Augmentation summary
        """
        self.logger.info(f"Augmenting {len(dataset_dirs)} existing datasets")
        
        from ga_components.file_analyzer import FileAnalyzer
        analyzer = FileAnalyzer()
        
        augmented_info = {}
        
        for dataset_dir in dataset_dirs:
            dataset_path = Path(dataset_dir)
            if not dataset_path.exists():
                self.logger.warning(f"Dataset directory not found: {dataset_dir}")
                continue
            
            self.logger.info(f"Processing existing dataset: {dataset_dir}")
            
            # Analyze all files in the dataset
            files_info = []
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        characteristics = analyzer.analyze_file(str(file_path))
                        files_info.append({
                            'file_path': str(file_path.relative_to(dataset_path)),
                            'characteristics': characteristics.to_dict()
                        })
                    except Exception as e:
                        self.logger.debug(f"Failed to analyze {file_path}: {e}")
            
            # Create augmented metadata
            dataset_name = dataset_path.name
            augmented_info[dataset_name] = {
                'original_path': str(dataset_path),
                'file_count': len(files_info),
                'files_analyzed': files_info,
                'domain_distribution': self._analyze_domain_distribution(files_info),
                'size_distribution': self._analyze_size_distribution(files_info)
            }
            
            # Save dataset-specific metadata
            metadata_file = dataset_path / f"{dataset_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(augmented_info[dataset_name], f, indent=2, default=str)
        
        return augmented_info
    
    def _define_dataset_specifications(self) -> Dict[str, List[DatasetSpec]]:
        """Define specifications for different domain datasets."""
        return {
            'text_data': [
                DatasetSpec(
                    domain='natural_language',
                    data_type='plain_text',
                    file_count=50,
                    size_range=(1024, 1024*1024),  # 1KB to 1MB
                    characteristics={
                        'language': 'english',
                        'text_type': 'mixed',
                        'repetition_level': 'medium'
                    }
                ),
                DatasetSpec(
                    domain='natural_language',
                    data_type='log_files',
                    file_count=20,
                    size_range=(10*1024, 10*1024*1024),  # 10KB to 10MB
                    characteristics={
                        'log_type': 'application',
                        'timestamp_format': 'iso',
                        'repetition_level': 'high'
                    }
                ),
                DatasetSpec(
                    domain='markup',
                    data_type='xml_documents',
                    file_count=30,
                    size_range=(5*1024, 5*1024*1024),  # 5KB to 5MB
                    characteristics={
                        'xml_type': 'structured_data',
                        'nesting_level': 'medium',
                        'repetition_level': 'high'
                    }
                )
            ],
            
            'source_code': [
                DatasetSpec(
                    domain='source_code',
                    data_type='python_files',
                    file_count=40,
                    size_range=(1024, 500*1024),  # 1KB to 500KB
                    characteristics={
                        'language': 'python',
                        'code_style': 'standard',
                        'comment_ratio': 0.2
                    }
                ),
                DatasetSpec(
                    domain='source_code',
                    data_type='javascript_files',
                    file_count=35,
                    size_range=(1024, 300*1024),  # 1KB to 300KB
                    characteristics={
                        'language': 'javascript',
                        'minified': False,
                        'comment_ratio': 0.15
                    }
                ),
                DatasetSpec(
                    domain='configuration',
                    data_type='json_configs',
                    file_count=25,
                    size_range=(512, 100*1024),  # 512B to 100KB
                    characteristics={
                        'config_type': 'application_config',
                        'nesting_level': 'medium',
                        'repetition_level': 'medium'
                    }
                )
            ],
            
            'scientific_data': [
                DatasetSpec(
                    domain='chemical_data',
                    data_type='sdf_files',
                    file_count=15,
                    size_range=(10*1024, 50*1024*1024),  # 10KB to 50MB
                    characteristics={
                        'compound_count': 'variable',
                        'structure_complexity': 'medium',
                        'properties_included': True
                    }
                ),
                DatasetSpec(
                    domain='numerical_data',
                    data_type='csv_datasets',
                    file_count=25,
                    size_range=(5*1024, 20*1024*1024),  # 5KB to 20MB
                    characteristics={
                        'data_type': 'numerical_measurements',
                        'column_count': 'variable',
                        'repetition_level': 'low'
                    }
                ),
                DatasetSpec(
                    domain='genomic_data',
                    data_type='fasta_sequences',
                    file_count=10,
                    size_range=(1024, 100*1024*1024),  # 1KB to 100MB
                    characteristics={
                        'sequence_type': 'dna',
                        'sequence_length': 'variable',
                        'repetition_level': 'very_high'
                    }
                )
            ],
            
            'binary_data': [
                DatasetSpec(
                    domain='binary',
                    data_type='random_binary',
                    file_count=20,
                    size_range=(1024, 10*1024*1024),  # 1KB to 10MB
                    characteristics={
                        'entropy': 'high',
                        'pattern': 'random',
                        'compressibility': 'low'
                    }
                ),
                DatasetSpec(
                    domain='binary',
                    data_type='structured_binary',
                    file_count=15,
                    size_range=(1024, 5*1024*1024),  # 1KB to 5MB
                    characteristics={
                        'entropy': 'medium',
                        'pattern': 'structured',
                        'compressibility': 'medium'
                    }
                )
            ],
            
            'mixed_data': [
                DatasetSpec(
                    domain='mixed',
                    data_type='archive_simulation',
                    file_count=10,
                    size_range=(1024*1024, 100*1024*1024),  # 1MB to 100MB
                    characteristics={
                        'content_mix': 'text_and_binary',
                        'structure': 'nested',
                        'compressibility': 'variable'
                    }
                )
            ]
        }
    
    def _generate_dataset(self, spec: DatasetSpec, output_dir: Path) -> Dict[str, Any]:
        """Generate a dataset according to the specification."""
        spec_dir = output_dir / spec.data_type
        spec_dir.mkdir(exist_ok=True)
        
        generated_files = []
        
        for i in range(spec.file_count):
            file_size = random.randint(spec.size_range[0], spec.size_range[1])
            
            # Generate filename
            if spec.data_type == 'python_files':
                filename = f"module_{i:03d}.py"
            elif spec.data_type == 'javascript_files':
                filename = f"script_{i:03d}.js"
            elif spec.data_type == 'json_configs':
                filename = f"config_{i:03d}.json"
            elif spec.data_type == 'xml_documents':
                filename = f"document_{i:03d}.xml"
            elif spec.data_type == 'sdf_files':
                filename = f"compounds_{i:03d}.sdf"
            elif spec.data_type == 'csv_datasets':
                filename = f"dataset_{i:03d}.csv"
            elif spec.data_type == 'fasta_sequences':
                filename = f"sequences_{i:03d}.fasta"
            elif spec.data_type == 'log_files':
                filename = f"application_{i:03d}.log"
            elif spec.data_type == 'plain_text':
                filename = f"text_{i:03d}.txt"
            elif spec.data_type == 'random_binary':
                filename = f"random_{i:03d}.bin"
            elif spec.data_type == 'structured_binary':
                filename = f"structured_{i:03d}.bin"
            elif spec.data_type == 'archive_simulation':
                filename = f"archive_{i:03d}.dat"
            else:
                filename = f"file_{i:03d}.data"
            
            file_path = spec_dir / filename
            
            # Generate file content
            self._generate_file_content(file_path, spec, file_size)
            generated_files.append(str(file_path.relative_to(output_dir)))
        
        return {
            'specification': spec.to_dict(),
            'output_directory': str(spec_dir),
            'file_count': len(generated_files),
            'files_created': generated_files
        }
    
    def _generate_file_content(self, file_path: Path, spec: DatasetSpec, target_size: int):
        """Generate content for a specific file type."""
        if spec.data_type == 'plain_text':
            self._generate_plain_text(file_path, target_size)
        elif spec.data_type == 'python_files':
            self._generate_python_code(file_path, target_size)
        elif spec.data_type == 'javascript_files':
            self._generate_javascript_code(file_path, target_size)
        elif spec.data_type == 'json_configs':
            self._generate_json_config(file_path, target_size)
        elif spec.data_type == 'xml_documents':
            self._generate_xml_document(file_path, target_size)
        elif spec.data_type == 'sdf_files':
            self._generate_sdf_file(file_path, target_size)
        elif spec.data_type == 'csv_datasets':
            self._generate_csv_dataset(file_path, target_size)
        elif spec.data_type == 'fasta_sequences':
            self._generate_fasta_sequences(file_path, target_size)
        elif spec.data_type == 'log_files':
            self._generate_log_file(file_path, target_size)
        elif spec.data_type == 'random_binary':
            self._generate_random_binary(file_path, target_size)
        elif spec.data_type == 'structured_binary':
            self._generate_structured_binary(file_path, target_size)
        elif spec.data_type == 'archive_simulation':
            self._generate_archive_simulation(file_path, target_size)
        else:
            # Default: random text
            self._generate_plain_text(file_path, target_size)
    
    def _generate_plain_text(self, file_path: Path, target_size: int):
        """Generate plain text with natural language characteristics."""
        # Common English words for more realistic text
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'can', 'said', 'each', 'which', 'time', 'data', 'system', 'computer',
            'program', 'algorithm', 'compression', 'performance', 'analysis'
        ]
        
        content = []
        current_size = 0
        
        while current_size < target_size:
            # Generate paragraphs
            paragraph_length = random.randint(50, 200)
            paragraph = []
            
            for _ in range(paragraph_length):
                if current_size >= target_size:
                    break
                
                word = random.choice(common_words)
                if len(paragraph) == 0:
                    word = word.capitalize()
                
                paragraph.append(word)
                current_size += len(word) + 1  # +1 for space
            
            if paragraph:
                paragraph_text = ' '.join(paragraph) + '.\n\n'
                content.append(paragraph_text)
                current_size += 2  # For the newlines
        
        with open(file_path, 'w') as f:
            f.write(''.join(content)[:target_size])
    
    def _generate_python_code(self, file_path: Path, target_size: int):
        """Generate synthetic Python code."""
        template = '''"""
Generated Python module for compression testing.
This module contains various functions and classes for demonstration.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional


class DataProcessor:
    """Example data processing class."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data = []
        self.processed = False
    
    def load_data(self, file_path: str) -> bool:
        """Load data from file."""
        try:
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def process_data(self) -> List[Dict]:
        """Process loaded data."""
        if not self.data:
            return []
        
        processed = []
        for item in self.data:
            if isinstance(item, dict):
                # Apply various processing steps
                processed_item = {
                    'id': item.get('id', 0),
                    'value': item.get('value', 0) * 2,
                    'category': item.get('category', 'unknown'),
                    'timestamp': item.get('timestamp', ''),
                    'processed': True
                }
                processed.append(processed_item)
        
        self.processed = True
        return processed
    
    def save_results(self, output_path: str, data: List[Dict]) -> bool:
        """Save processed results."""
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for numerical data."""
    if not data:
        return {}
    
    data_sorted = sorted(data)
    n = len(data)
    
    stats = {
        'count': n,
        'min': min(data),
        'max': max(data),
        'mean': sum(data) / n,
        'median': data_sorted[n // 2],
        'sum': sum(data)
    }
    
    # Calculate variance and standard deviation
    mean = stats['mean']
    variance = sum((x - mean) ** 2 for x in data) / n
    stats['variance'] = variance
    stats['std_dev'] = variance ** 0.5
    
    return stats


def generate_test_data(count: int = 1000) -> List[Dict]:
    """Generate test data for processing."""
    data = []
    categories = ['A', 'B', 'C', 'D', 'E']
    
    for i in range(count):
        item = {
            'id': i,
            'value': random.uniform(0, 100),
            'category': random.choice(categories),
            'timestamp': f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            'metadata': {
                'source': 'generated',
                'version': '1.0',
                'flags': [random.choice(['flag1', 'flag2', 'flag3']) for _ in range(random.randint(1, 3))]
            }
        }
        data.append(item)
    
    return data


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor({'debug': True})
    test_data = generate_test_data(500)
    
    # Process data
    results = processor.process_data()
    
    # Calculate statistics
    values = [item['value'] for item in test_data if 'value' in item]
    stats = calculate_statistics(values)
    
    print(f"Processed {len(results)} items")
    print(f"Statistics: {stats}")
'''
        
        # Repeat and modify template to reach target size
        content = template
        while len(content) < target_size:
            # Add more functions or modify existing ones
            additional = f'''

def utility_function_{random.randint(1000, 9999)}(param1, param2=None):
    """Utility function for demonstration."""
    result = []
    for i in range(param1):
        if param2:
            result.append(i * param2)
        else:
            result.append(i ** 2)
    return result

'''
            content += additional
        
        with open(file_path, 'w') as f:
            f.write(content[:target_size])
    
    def _generate_javascript_code(self, file_path: Path, target_size: int):
        """Generate synthetic JavaScript code."""
        template = '''/**
 * Generated JavaScript module for compression testing
 * Contains various functions and classes for demonstration
 */

class DataManager {
    constructor(config = {}) {
        this.config = config;
        this.data = [];
        this.processed = false;
    }
    
    async loadData(filePath) {
        try {
            const response = await fetch(filePath);
            this.data = await response.json();
            return true;
        } catch (error) {
            console.error('Error loading data:', error);
            return false;
        }
    }
    
    processData() {
        if (!this.data.length) return [];
        
        return this.data.map(item => ({
            id: item.id || 0,
            value: (item.value || 0) * 2,
            category: item.category || 'unknown',
            timestamp: item.timestamp || '',
            processed: true
        }));
    }
    
    saveResults(data) {
        const jsonData = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonData], { type: 'application/json' });
        return blob;
    }
}

function calculateStatistics(data) {
    if (!Array.isArray(data) || !data.length) return {};
    
    const sorted = [...data].sort((a, b) => a - b);
    const n = data.length;
    const sum = data.reduce((acc, val) => acc + val, 0);
    const mean = sum / n;
    
    return {
        count: n,
        min: Math.min(...data),
        max: Math.max(...data),
        mean: mean,
        median: sorted[Math.floor(n / 2)],
        sum: sum,
        variance: data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n
    };
}

function generateTestData(count = 1000) {
    const categories = ['A', 'B', 'C', 'D', 'E'];
    const data = [];
    
    for (let i = 0; i < count; i++) {
        data.push({
            id: i,
            value: Math.random() * 100,
            category: categories[Math.floor(Math.random() * categories.length)],
            timestamp: new Date().toISOString(),
            metadata: {
                source: 'generated',
                version: '1.0',
                flags: Array.from({ length: Math.floor(Math.random() * 3) + 1 }, 
                    () => 'flag' + Math.floor(Math.random() * 3 + 1))
            }
        });
    }
    
    return data;
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DataManager, calculateStatistics, generateTestData };
}
'''
        
        content = template
        while len(content) < target_size:
            additional = f'''

function utilityFunction{random.randint(1000, 9999)}(param1, param2 = null) {{
    const result = [];
    for (let i = 0; i < param1; i++) {{
        if (param2) {{
            result.push(i * param2);
        }} else {{
            result.push(i ** 2);
        }}
    }}
    return result;
}}

'''
            content += additional
        
        with open(file_path, 'w') as f:
            f.write(content[:target_size])
    
    def _generate_json_config(self, file_path: Path, target_size: int):
        """Generate JSON configuration files."""
        config = {
            "application": {
                "name": "TestApplication",
                "version": "1.0.0",
                "debug": True,
                "log_level": "INFO"
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
                "pool_size": 10,
                "timeout": 30
            },
            "cache": {
                "enabled": True,
                "type": "redis",
                "host": "localhost",
                "port": 6379,
                "ttl": 3600
            },
            "features": {
                "enable_analytics": True,
                "enable_logging": True,
                "enable_metrics": True,
                "enable_profiling": False
            },
            "api": {
                "base_url": "https://api.example.com",
                "timeout": 30,
                "retries": 3,
                "rate_limit": 100
            }
        }
        
        # Expand config to reach target size
        while len(json.dumps(config)) < target_size:
            service_num = random.randint(1, 1000)
            config[f"service_{service_num}"] = {
                "enabled": random.choice([True, False]),
                "endpoint": f"https://service{service_num}.example.com",
                "timeout": random.randint(10, 60),
                "retries": random.randint(1, 5),
                "config": {
                    "param1": random.randint(1, 100),
                    "param2": random.choice(["option1", "option2", "option3"]),
                    "param3": [random.randint(1, 10) for _ in range(5)]
                }
            }
        
        content = json.dumps(config, indent=2)[:target_size]
        with open(file_path, 'w') as f:
            f.write(content)
    
    def _generate_xml_document(self, file_path: Path, target_size: int):
        """Generate XML documents."""
        root = ET.Element("dataset")
        root.set("version", "1.0")
        root.set("generated", datetime.now().isoformat())
        
        current_size = len(ET.tostring(root).decode())
        
        while current_size < target_size:
            record = ET.SubElement(root, "record")
            record.set("id", str(random.randint(1, 10000)))
            
            # Add various elements
            name_elem = ET.SubElement(record, "name")
            name_elem.text = f"Item_{random.randint(1, 1000)}"
            
            value_elem = ET.SubElement(record, "value")
            value_elem.text = str(random.uniform(0, 100))
            
            category_elem = ET.SubElement(record, "category")
            category_elem.text = random.choice(["A", "B", "C", "D", "E"])
            
            metadata = ET.SubElement(record, "metadata")
            for i in range(random.randint(1, 5)):
                prop = ET.SubElement(metadata, f"property_{i}")
                prop.text = f"value_{random.randint(1, 100)}"
            
            current_size = len(ET.tostring(root).decode())
            if current_size > target_size:
                break
        
        # Write XML to file
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='unicode', xml_declaration=True)
    
    def _generate_sdf_file(self, file_path: Path, target_size: int):
        """Generate SDF (Structure Data Format) files for chemical data."""
        current_size = 0
        compound_id = 1
        
        with open(file_path, 'w') as f:
            while current_size < target_size:
                # Simple SDF structure simulation
                compound_data = f"""Compound_{compound_id:04d}
  Generated molecule for testing
  
 10 10  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    1.7320    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.7320    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5000    0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  5  6  1  0  0  0  0
  6  1  1  0  0  0  0
M  END
> <MOLECULAR_WEIGHT>
{random.uniform(100, 500):.2f}

> <FORMULA>
C{random.randint(5, 20)}H{random.randint(10, 40)}N{random.randint(0, 5)}O{random.randint(0, 8)}

> <ACTIVITY>
{random.uniform(0, 1):.6f}

$$$$
"""
                f.write(compound_data)
                current_size += len(compound_data)
                compound_id += 1
                
                if current_size >= target_size:
                    break
    
    def _generate_csv_dataset(self, file_path: Path, target_size: int):
        """Generate CSV datasets with numerical data."""
        headers = ['id', 'timestamp', 'value1', 'value2', 'value3', 'category', 'status']
        categories = ['A', 'B', 'C', 'D', 'E']
        statuses = ['active', 'inactive', 'pending', 'completed']
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            current_size = len(','.join(headers)) + 1  # +1 for newline
            row_id = 1
            
            while current_size < target_size:
                row = [
                    row_id,
                    datetime.now() + timedelta(minutes=row_id),
                    round(random.uniform(0, 100), 3),
                    round(random.uniform(-50, 50), 3),
                    round(random.uniform(0, 1000), 2),
                    random.choice(categories),
                    random.choice(statuses)
                ]
                
                writer.writerow(row)
                current_size += sum(len(str(cell)) for cell in row) + len(row) - 1  # -1 for commas
                row_id += 1
                
                if current_size >= target_size:
                    break
    
    def _generate_fasta_sequences(self, file_path: Path, target_size: int):
        """Generate FASTA sequence files for genomic data."""
        bases = ['A', 'T', 'G', 'C']
        current_size = 0
        seq_id = 1
        
        with open(file_path, 'w') as f:
            while current_size < target_size:
                header = f">Sequence_{seq_id:04d} | Generated DNA sequence for testing\n"
                f.write(header)
                current_size += len(header)
                
                # Generate DNA sequence (highly repetitive for compression testing)
                seq_length = random.randint(100, 10000)
                sequence = ""
                
                # Create patterns for better compression
                pattern = ''.join(random.choices(bases, k=random.randint(10, 50)))
                repeats = seq_length // len(pattern)
                remainder = seq_length % len(pattern)
                
                sequence = pattern * repeats + pattern[:remainder]
                
                # Write sequence in lines of 80 characters
                for i in range(0, len(sequence), 80):
                    line = sequence[i:i+80] + '\n'
                    f.write(line)
                    current_size += len(line)
                    
                    if current_size >= target_size:
                        break
                
                seq_id += 1
                if current_size >= target_size:
                    break
    
    def _generate_log_file(self, file_path: Path, target_size: int):
        """Generate application log files."""
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        components = ['Database', 'API', 'Cache', 'Auth', 'Worker', 'Scheduler']
        messages = [
            'Connection established successfully',
            'Processing request',
            'Cache miss, fetching from database',
            'Authentication successful',
            'Task completed',
            'Scheduled job executed',
            'Connection timeout',
            'Invalid request format',
            'Database query failed',
            'Memory usage warning'
        ]
        
        current_size = 0
        base_time = datetime.now()
        
        with open(file_path, 'w') as f:
            entry_id = 1
            while current_size < target_size:
                timestamp = (base_time + timedelta(seconds=entry_id)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                level = random.choice(log_levels)
                component = random.choice(components)
                message = random.choice(messages)
                
                log_entry = f"[{timestamp}] {level:8} [{component}] {message} (ID: {entry_id:06d})\n"
                f.write(log_entry)
                current_size += len(log_entry)
                entry_id += 1
                
                if current_size >= target_size:
                    break
    
    def _generate_random_binary(self, file_path: Path, target_size: int):
        """Generate random binary data (high entropy)."""
        with open(file_path, 'wb') as f:
            f.write(os.urandom(target_size))
    
    def _generate_structured_binary(self, file_path: Path, target_size: int):
        """Generate structured binary data (medium entropy)."""
        with open(file_path, 'wb') as f:
            pattern = os.urandom(1024)  # 1KB pattern
            written = 0
            
            while written < target_size:
                chunk_size = min(len(pattern), target_size - written)
                f.write(pattern[:chunk_size])
                written += chunk_size
    
    def _generate_archive_simulation(self, file_path: Path, target_size: int):
        """Generate files that simulate archive contents (mixed data)."""
        with open(file_path, 'wb') as f:
            written = 0
            
            while written < target_size:
                # Alternate between text-like and binary-like sections
                if random.choice([True, False]):
                    # Text-like section
                    text_data = ' '.join(random.choices(
                        ['data', 'file', 'system', 'process', 'result', 'value', 'item'],
                        k=random.randint(100, 1000)
                    )).encode('utf-8')
                    chunk = text_data
                else:
                    # Binary-like section  
                    chunk = os.urandom(random.randint(100, 5000))
                
                chunk_size = min(len(chunk), target_size - written)
                f.write(chunk[:chunk_size])
                written += chunk_size
    
    def _create_dataset_metadata(self, created_datasets: Dict) -> Dict[str, Any]:
        """Create comprehensive metadata for the dataset collection."""
        total_files = 0
        total_size = 0
        domain_summary = {}
        
        for domain, datasets in created_datasets.items():
            domain_files = 0
            domain_size = 0
            
            for dataset in datasets:
                dataset_files = dataset['file_count']
                domain_files += dataset_files
                total_files += dataset_files
                
                # Calculate total size (approximate)
                for file_path in dataset['files_created']:
                    full_path = self.output_base_dir / domain / file_path
                    if full_path.exists():
                        size = full_path.stat().st_size
                        domain_size += size
                        total_size += size
            
            domain_summary[domain] = {
                'file_count': domain_files,
                'total_size_bytes': domain_size,
                'datasets': len(datasets)
            }
        
        return {
            'collection_metadata': {
                'creation_date': datetime.now().isoformat(),
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'domains_count': len(created_datasets),
                'base_directory': str(self.output_base_dir)
            },
            'domain_summary': domain_summary,
            'detailed_datasets': created_datasets
        }
    
    def _create_dataset_readme(self, metadata: Dict):
        """Create README file for the dataset collection."""
        readme_content = f"""# Multi-Domain Dataset Collection

Generated on: {metadata['collection_metadata']['creation_date']}

## Collection Summary

- **Total Files**: {metadata['collection_metadata']['total_files']:,}
- **Total Size**: {metadata['collection_metadata']['total_size_mb']:.2f} MB
- **Domains**: {metadata['collection_metadata']['domains_count']}
- **Base Directory**: {metadata['collection_metadata']['base_directory']}

## Domain Breakdown

"""
        
        for domain, summary in metadata['domain_summary'].items():
            readme_content += f"""### {domain.replace('_', ' ').title()}
- Files: {summary['file_count']:,}
- Size: {summary['total_size_bytes'] / (1024 * 1024):.2f} MB
- Dataset Types: {summary['datasets']}

"""
        
        readme_content += """## Usage

This dataset collection is designed for comprehensive multi-domain compression testing.
Each domain contains files with different characteristics to evaluate compressor
performance across various data types.

### Recommended Usage

1. **Single Domain Testing**: Test compressors on individual domains
2. **Cross-Domain Evaluation**: Compare performance across different data types
3. **Batch Processing**: Use the entire collection for comprehensive benchmarking

### File Types Included

- **Text Data**: Natural language text, logs, markup documents
- **Source Code**: Python, JavaScript, configuration files
- **Scientific Data**: Chemical compounds, numerical datasets, genomic sequences  
- **Binary Data**: Random and structured binary files
- **Mixed Data**: Archive-like files with varied content

## Dataset Generation

This collection was generated using synthetic data creation algorithms designed to
simulate real-world file characteristics while providing controlled test conditions
for compression algorithm evaluation.
"""
        
        readme_file = self.output_base_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
    
    def _analyze_domain_distribution(self, files_info: List[Dict]) -> Dict[str, int]:
        """Analyze domain distribution in file collection."""
        domain_counts = {}
        for file_info in files_info:
            domain = file_info['characteristics']['data_domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return domain_counts
    
    def _analyze_size_distribution(self, files_info: List[Dict]) -> Dict[str, Any]:
        """Analyze file size distribution."""
        sizes = [file_info['characteristics']['file_size'] for file_info in files_info]
        if not sizes:
            return {}
        
        return {
            'total_files': len(sizes),
            'total_size_bytes': sum(sizes),
            'average_size_bytes': sum(sizes) / len(sizes),
            'min_size_bytes': min(sizes),
            'max_size_bytes': max(sizes),
            'size_ranges': {
                'small_files_1kb': len([s for s in sizes if s <= 1024]),
                'medium_files_1mb': len([s for s in sizes if 1024 < s <= 1024*1024]),
                'large_files_10mb': len([s for s in sizes if 1024*1024 < s <= 10*1024*1024]),
                'very_large_files': len([s for s in sizes if s > 10*1024*1024])
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the dataset collector."""
        logger = logging.getLogger('DatasetCollector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Domain Dataset Collector')
    parser.add_argument('--output-dir', default='datasets', 
                       help='Output directory for datasets (default: datasets)')
    parser.add_argument('--create-collection', action='store_true',
                       help='Create comprehensive dataset collection')
    parser.add_argument('--augment-datasets', nargs='+',
                       help='Augment existing dataset directories')
    
    args = parser.parse_args()
    
    collector = MultiDomainDatasetCollector(args.output_dir)
    
    if args.create_collection:
        print("Creating comprehensive dataset collection...")
        metadata = collector.create_comprehensive_dataset_collection()
        print(f"Dataset collection created successfully!")
        print(f"Total files: {metadata['collection_metadata']['total_files']}")
        print(f"Total size: {metadata['collection_metadata']['total_size_mb']:.2f} MB")
        print(f"Output directory: {metadata['collection_metadata']['base_directory']}")
    
    if args.augment_datasets:
        print(f"Augmenting {len(args.augment_datasets)} existing datasets...")
        results = collector.augment_existing_datasets(args.augment_datasets)
        print("Augmentation completed!")
        for dataset, info in results.items():
            print(f"  {dataset}: {info['file_count']} files analyzed")


if __name__ == "__main__":
    main()