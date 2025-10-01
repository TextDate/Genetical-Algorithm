"""
Cross-Domain Performance Evaluation

Provides specialized metrics and evaluation methods for comparing compressor
performance across different data domains and types.
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import statistics

from ga_logging import get_logger
from ga_components.file_analyzer import DataType, DataDomain


class MetricType(Enum):
    """Types of evaluation metrics."""
    COMPRESSION_EFFICIENCY = "compression_efficiency"
    TIME_EFFICIENCY = "time_efficiency"
    MEMORY_EFFICIENCY = "memory_efficiency"
    OVERALL_PERFORMANCE = "overall_performance"
    DOMAIN_SPECIFICITY = "domain_specificity"
    CONSISTENCY = "consistency"
    SCALABILITY = "scalability"


@dataclass
class CrossDomainMetrics:
    """Cross-domain evaluation metrics."""
    # Core performance metrics
    weighted_compression_score: float
    time_normalized_score: float
    memory_efficiency_score: float
    overall_performance_index: float
    
    # Cross-domain specific metrics
    domain_consistency_score: float
    data_type_adaptability: float
    prediction_accuracy_rate: float
    
    # Comparative metrics
    relative_performance_rank: int
    domain_leadership_count: int  # Number of domains where this compressor leads
    versatility_index: float  # How well it performs across different domains
    
    # Statistical metrics
    performance_variance: float
    performance_stability: float
    outlier_resistance: float


class CrossDomainEvaluator:
    """
    Advanced evaluation system for cross-domain compressor performance analysis.
    
    Implements sophisticated metrics that account for different data characteristics,
    domain-specific requirements, and comparative performance analysis.
    """
    
    # Domain-specific performance weights
    DOMAIN_WEIGHTS = {
        DataDomain.NATURAL_LANGUAGE: {'compression': 0.6, 'time': 0.2, 'memory': 0.2},
        DataDomain.SOURCE_CODE: {'compression': 0.5, 'time': 0.3, 'memory': 0.2},
        DataDomain.CHEMICAL_DATA: {'compression': 0.7, 'time': 0.2, 'memory': 0.1},
        DataDomain.GENOMIC_DATA: {'compression': 0.8, 'time': 0.1, 'memory': 0.1},
        DataDomain.NUMERICAL_DATA: {'compression': 0.6, 'time': 0.3, 'memory': 0.1},
        DataDomain.IMAGE_DATA: {'compression': 0.4, 'time': 0.4, 'memory': 0.2},
        DataDomain.LOG_DATA: {'compression': 0.7, 'time': 0.2, 'memory': 0.1},
        DataDomain.DATABASE_FILE: {'compression': 0.6, 'time': 0.3, 'memory': 0.1},
        DataDomain.CONFIGURATION: {'compression': 0.5, 'time': 0.3, 'memory': 0.2}
    }
    
    # Data type base expectations (baseline compression ratios)
    TYPE_BASELINES = {
        DataType.TEXT: 3.0,
        DataType.CODE: 4.0, 
        DataType.SCIENTIFIC: 5.0,
        DataType.BINARY: 1.8,
        DataType.MULTIMEDIA: 1.2,
        DataType.DATABASE: 3.5,
        DataType.COMPRESSED: 1.05
    }
    
    def __init__(self):
        self.logger = get_logger("CrossDomainEvaluator")
    
    def evaluate_cross_domain_performance(self, benchmark_results: List[Dict]) -> Dict[str, CrossDomainMetrics]:
        """
        Evaluate cross-domain performance for all compressors.
        
        Args:
            benchmark_results: List of benchmark result dictionaries
            
        Returns:
            Dictionary mapping compressor names to CrossDomainMetrics
        """
        self.logger.info("Starting cross-domain performance evaluation")
        
        if not benchmark_results:
            return {}
        
        # Group results by compressor
        compressor_results = self._group_by_compressor(benchmark_results)
        
        # Calculate metrics for each compressor
        compressor_metrics = {}
        for compressor, results in compressor_results.items():
            metrics = self._calculate_compressor_metrics(compressor, results, benchmark_results)
            compressor_metrics[compressor] = metrics
        
        self.logger.info(f"Evaluated {len(compressor_metrics)} compressors across domains")
        return compressor_metrics
    
    def _group_by_compressor(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group benchmark results by compressor."""
        grouped = {}
        for result in results:
            compressor = result['compressor_name']
            if compressor not in grouped:
                grouped[compressor] = []
            grouped[compressor].append(result)
        return grouped
    
    def _calculate_compressor_metrics(self, compressor_name: str, 
                                    compressor_results: List[Dict],
                                    all_results: List[Dict]) -> CrossDomainMetrics:
        """Calculate comprehensive metrics for a single compressor."""
        
        # Core performance calculations
        weighted_compression = self._calculate_weighted_compression_score(compressor_results)
        time_normalized = self._calculate_time_normalized_score(compressor_results)
        memory_efficiency = self._calculate_memory_efficiency_score(compressor_results)
        overall_performance = self._calculate_overall_performance_index(
            weighted_compression, time_normalized, memory_efficiency
        )
        
        # Cross-domain specific metrics
        domain_consistency = self._calculate_domain_consistency_score(compressor_results)
        data_type_adaptability = self._calculate_data_type_adaptability(compressor_results)
        prediction_accuracy = self._calculate_prediction_accuracy_rate(compressor_results)
        
        # Comparative metrics
        relative_rank = self._calculate_relative_performance_rank(
            compressor_name, compressor_results, all_results
        )
        domain_leadership = self._calculate_domain_leadership_count(
            compressor_name, all_results
        )
        versatility = self._calculate_versatility_index(compressor_results)
        
        # Statistical metrics
        performance_variance = self._calculate_performance_variance(compressor_results)
        performance_stability = self._calculate_performance_stability(compressor_results)
        outlier_resistance = self._calculate_outlier_resistance(compressor_results)
        
        return CrossDomainMetrics(
            weighted_compression_score=weighted_compression,
            time_normalized_score=time_normalized,
            memory_efficiency_score=memory_efficiency,
            overall_performance_index=overall_performance,
            domain_consistency_score=domain_consistency,
            data_type_adaptability=data_type_adaptability,
            prediction_accuracy_rate=prediction_accuracy,
            relative_performance_rank=relative_rank,
            domain_leadership_count=domain_leadership,
            versatility_index=versatility,
            performance_variance=performance_variance,
            performance_stability=performance_stability,
            outlier_resistance=outlier_resistance
        )
    
    def _calculate_weighted_compression_score(self, results: List[Dict]) -> float:
        """Calculate domain-weighted compression performance score."""
        if not results:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in results:
            domain = DataDomain(result['file_characteristics']['data_domain'])
            data_type = DataType(result['file_characteristics']['data_type'])
            compression_ratio = result['best_compression_ratio']
            
            # Get domain weight for compression
            domain_weight = self.DOMAIN_WEIGHTS.get(domain, {'compression': 0.6})['compression']
            
            # Normalize against expected baseline for data type
            baseline = self.TYPE_BASELINES.get(data_type, 2.0)
            normalized_ratio = min(compression_ratio / baseline, 2.0)  # Cap at 2x expected
            
            # Apply domain weighting
            weighted_score = normalized_ratio * domain_weight
            total_weighted_score += weighted_score
            total_weight += domain_weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_time_normalized_score(self, results: List[Dict]) -> float:
        """Calculate time efficiency score normalized across domains."""
        if not results:
            return 0.0
        
        valid_times = [r['best_compression_time'] for r in results 
                      if r['best_compression_time'] > 0]
        
        if not valid_times:
            return 0.0
        
        # Calculate efficiency as compression_ratio / time
        efficiencies = []
        for result in results:
            if result['best_compression_time'] > 0:
                efficiency = result['best_compression_ratio'] / result['best_compression_time']
                efficiencies.append(efficiency)
        
        return statistics.mean(efficiencies) if efficiencies else 0.0
    
    def _calculate_memory_efficiency_score(self, results: List[Dict]) -> float:
        """Calculate memory efficiency score."""
        if not results:
            return 0.0
        
        # Use RAM usage if available, otherwise estimate based on file size and compression ratio
        memory_efficiencies = []
        
        for result in results:
            ram_usage = result.get('best_ram_usage', 0)
            if ram_usage > 0:
                # Memory efficiency = compression_ratio / (ram_usage_mb / file_size_mb)
                file_size_mb = result['file_characteristics']['file_size'] / (1024 * 1024)
                ram_ratio = ram_usage / max(file_size_mb, 0.1)  # Avoid division by zero
                memory_efficiency = result['best_compression_ratio'] / max(ram_ratio, 0.1)
                memory_efficiencies.append(memory_efficiency)
        
        return statistics.mean(memory_efficiencies) if memory_efficiencies else 1.0
    
    def _calculate_overall_performance_index(self, compression_score: float,
                                           time_score: float, 
                                           memory_score: float) -> float:
        """Calculate overall performance index combining all factors."""
        # Weighted geometric mean to avoid one poor metric dominating
        weights = [0.5, 0.3, 0.2]  # compression, time, memory
        scores = [max(compression_score, 0.1), max(time_score, 0.1), max(memory_score, 0.1)]
        
        # Geometric mean with weights
        product = 1.0
        for score, weight in zip(scores, weights):
            product *= score ** weight
        
        return product
    
    def _calculate_domain_consistency_score(self, results: List[Dict]) -> float:
        """Calculate how consistently the compressor performs across domains."""
        if len(results) < 2:
            return 1.0
        
        # Group by domain
        domain_performance = {}
        for result in results:
            domain = result['file_characteristics']['data_domain']
            if domain not in domain_performance:
                domain_performance[domain] = []
            domain_performance[domain].append(result['best_compression_ratio'])
        
        # Calculate coefficient of variation across domains
        domain_averages = [statistics.mean(ratios) for ratios in domain_performance.values()]
        
        if len(domain_averages) < 2:
            return 1.0
        
        mean_performance = statistics.mean(domain_averages)
        std_performance = statistics.stdev(domain_averages)
        
        # Consistency score = 1 - coefficient_of_variation
        cv = std_performance / mean_performance if mean_performance > 0 else 1.0
        return max(0.0, 1.0 - cv)
    
    def _calculate_data_type_adaptability(self, results: List[Dict]) -> float:
        """Calculate how well the compressor adapts to different data types."""
        if not results:
            return 0.0
        
        type_performance = {}
        for result in results:
            data_type = result['file_characteristics']['data_type']
            if data_type not in type_performance:
                type_performance[data_type] = []
            
            # Normalize performance against type baseline
            baseline = self.TYPE_BASELINES.get(DataType(data_type), 2.0)
            normalized_perf = result['best_compression_ratio'] / baseline
            type_performance[data_type].append(normalized_perf)
        
        # Calculate average normalized performance across types
        type_averages = [statistics.mean(perfs) for perfs in type_performance.values()]
        return statistics.mean(type_averages) if type_averages else 0.0
    
    def _calculate_prediction_accuracy_rate(self, results: List[Dict]) -> float:
        """Calculate how accurately the file analyzer predicted compressibility."""
        if not results:
            return 0.0
        
        accurate_predictions = 0
        total_predictions = 0
        
        for result in results:
            predicted = result['file_characteristics']['predicted_compressibility']
            actual_ratio = result['best_compression_ratio']
            
            # Map actual ratio to category
            if actual_ratio >= 3.0:
                actual = "high"
            elif actual_ratio >= 2.0:
                actual = "medium"
            else:
                actual = "low"
            
            total_predictions += 1
            if predicted == actual:
                accurate_predictions += 1
        
        return accurate_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_relative_performance_rank(self, compressor_name: str,
                                           compressor_results: List[Dict],
                                           all_results: List[Dict]) -> int:
        """Calculate relative performance rank compared to other compressors."""
        # Calculate average compression ratio for this compressor
        comp_avg = statistics.mean([r['best_compression_ratio'] for r in compressor_results])
        
        # Calculate averages for all compressors
        compressor_averages = {}
        for result in all_results:
            comp = result['compressor_name']
            if comp not in compressor_averages:
                compressor_averages[comp] = []
            compressor_averages[comp].append(result['best_compression_ratio'])
        
        # Calculate final averages and rank
        final_averages = {comp: statistics.mean(ratios) 
                         for comp, ratios in compressor_averages.items()}
        
        sorted_compressors = sorted(final_averages.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        for rank, (comp, _) in enumerate(sorted_compressors, 1):
            if comp == compressor_name:
                return rank
        
        return len(sorted_compressors)
    
    def _calculate_domain_leadership_count(self, compressor_name: str,
                                         all_results: List[Dict]) -> int:
        """Count how many domains this compressor leads in."""
        # Group by domain
        domain_performance = {}
        for result in all_results:
            domain = result['file_characteristics']['data_domain']
            if domain not in domain_performance:
                domain_performance[domain] = {}
            
            comp = result['compressor_name']
            if comp not in domain_performance[domain]:
                domain_performance[domain][comp] = []
            
            domain_performance[domain][comp].append(result['best_compression_ratio'])
        
        # Count leadership
        leadership_count = 0
        for domain, comp_results in domain_performance.items():
            # Calculate averages for each compressor in this domain
            comp_averages = {comp: statistics.mean(ratios) 
                           for comp, ratios in comp_results.items()}
            
            # Find leader
            leader = max(comp_averages.items(), key=lambda x: x[1])[0]
            if leader == compressor_name:
                leadership_count += 1
        
        return leadership_count
    
    def _calculate_versatility_index(self, results: List[Dict]) -> float:
        """Calculate versatility index based on performance across different contexts."""
        if not results:
            return 0.0
        
        # Count unique domains and data types handled
        domains = set(r['file_characteristics']['data_domain'] for r in results)
        data_types = set(r['file_characteristics']['data_type'] for r in results)
        
        # Base versatility on coverage and performance consistency
        domain_coverage = len(domains) / len(DataDomain)
        type_coverage = len(data_types) / len(DataType)
        
        # Performance consistency (lower variance = higher versatility)
        ratios = [r['best_compression_ratio'] for r in results]
        consistency = 1.0 - (statistics.stdev(ratios) / statistics.mean(ratios)) if len(ratios) > 1 else 1.0
        consistency = max(0.0, consistency)
        
        # Combine factors
        versatility = (domain_coverage * 0.4 + type_coverage * 0.3 + consistency * 0.3)
        return versatility
    
    def _calculate_performance_variance(self, results: List[Dict]) -> float:
        """Calculate performance variance across all benchmarks."""
        if len(results) < 2:
            return 0.0
        
        ratios = [r['best_compression_ratio'] for r in results]
        return statistics.variance(ratios)
    
    def _calculate_performance_stability(self, results: List[Dict]) -> float:
        """Calculate performance stability (inverse of coefficient of variation)."""
        if len(results) < 2:
            return 1.0
        
        ratios = [r['best_compression_ratio'] for r in results]
        mean_ratio = statistics.mean(ratios)
        std_ratio = statistics.stdev(ratios)
        
        if mean_ratio == 0:
            return 0.0
        
        cv = std_ratio / mean_ratio
        return max(0.0, 1.0 - cv)
    
    def _calculate_outlier_resistance(self, results: List[Dict]) -> float:
        """Calculate resistance to outliers (robustness)."""
        if len(results) < 3:
            return 1.0
        
        ratios = [r['best_compression_ratio'] for r in results]
        
        # Compare median vs mean (outlier resistance indicator)
        median_ratio = statistics.median(ratios)
        mean_ratio = statistics.mean(ratios)
        
        if mean_ratio == 0:
            return 0.0
        
        # Smaller difference indicates better outlier resistance
        difference_ratio = abs(median_ratio - mean_ratio) / mean_ratio
        return max(0.0, 1.0 - difference_ratio)
    
    def generate_performance_report(self, compressor_metrics: Dict[str, CrossDomainMetrics]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not compressor_metrics:
            return {}
        
        # Overall rankings
        overall_ranking = sorted(
            compressor_metrics.items(),
            key=lambda x: x[1].overall_performance_index,
            reverse=True
        )
        
        # Best in category
        best_compression = max(compressor_metrics.items(),
                              key=lambda x: x[1].weighted_compression_score)
        best_speed = max(compressor_metrics.items(),
                        key=lambda x: x[1].time_normalized_score)
        best_memory = max(compressor_metrics.items(),
                         key=lambda x: x[1].memory_efficiency_score)
        most_versatile = max(compressor_metrics.items(),
                           key=lambda x: x[1].versatility_index)
        most_consistent = max(compressor_metrics.items(),
                            key=lambda x: x[1].domain_consistency_score)
        
        report = {
            'overall_ranking': [(name, metrics.overall_performance_index) 
                              for name, metrics in overall_ranking],
            'category_leaders': {
                'best_compression': (best_compression[0], best_compression[1].weighted_compression_score),
                'best_speed': (best_speed[0], best_speed[1].time_normalized_score),
                'best_memory': (best_memory[0], best_memory[1].memory_efficiency_score),
                'most_versatile': (most_versatile[0], most_versatile[1].versatility_index),
                'most_consistent': (most_consistent[0], most_consistent[1].domain_consistency_score)
            },
            'detailed_metrics': {name: {
                'overall_performance_index': metrics.overall_performance_index,
                'weighted_compression_score': metrics.weighted_compression_score,
                'time_normalized_score': metrics.time_normalized_score,
                'memory_efficiency_score': metrics.memory_efficiency_score,
                'domain_consistency_score': metrics.domain_consistency_score,
                'versatility_index': metrics.versatility_index,
                'relative_performance_rank': metrics.relative_performance_rank,
                'domain_leadership_count': metrics.domain_leadership_count
            } for name, metrics in compressor_metrics.items()}
        }
        
        return report