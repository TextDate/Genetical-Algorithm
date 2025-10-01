"""
Automatic Compressor Recommendation Engine

Provides intelligent compressor selection based on file characteristics,
historical performance data, and domain-specific optimization requirements.
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import statistics

from ga_logging import get_logger
from ga_components.file_analyzer import FileAnalyzer, FileCharacteristics, DataType, DataDomain


@dataclass
class RecommendationContext:
    """Context for compressor recommendation."""
    optimization_target: str  # 'ratio', 'speed', 'memory', 'balanced'
    max_time_budget: Optional[float] = None  # Maximum acceptable compression time
    memory_constraints: Optional[int] = None  # Maximum memory usage in MB
    file_size_threshold: Optional[int] = None  # Large file threshold
    quality_requirements: Optional[str] = None  # 'high', 'medium', 'low'


@dataclass
class CompressorRecommendation:
    """Recommendation result for a compressor."""
    compressor_name: str
    confidence_score: float  # 0.0 - 1.0
    expected_compression_ratio: float
    expected_compression_time: float
    expected_memory_usage: float
    reasoning: List[str]  # Human-readable explanation
    
    # Performance predictions with uncertainty
    ratio_range: Tuple[float, float]  # (min, max) expected
    time_range: Tuple[float, float]   # (min, max) expected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'compressor_name': self.compressor_name,
            'confidence_score': self.confidence_score,
            'expected_compression_ratio': self.expected_compression_ratio,
            'expected_compression_time': self.expected_compression_time,
            'expected_memory_usage': self.expected_memory_usage,
            'reasoning': self.reasoning,
            'ratio_range': self.ratio_range,
            'time_range': self.time_range
        }


class CompressorRecommender:
    """
    Intelligent compressor recommendation system using machine learning
    and heuristic approaches based on file characteristics and historical data.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.logger = get_logger("CompressorRecommender")
        self.file_analyzer = FileAnalyzer()
        self.knowledge_base_path = knowledge_base_path or "knowledge_base.json"
        self.historical_data: Dict[str, List] = {}
        self.domain_performance_cache: Dict[str, Dict] = {}
        
        # Load historical performance data if available
        self._load_knowledge_base()
        
        # Default performance expectations (will be updated from historical data)
        self.default_expectations = {
            'zstd': {
                'compression_factor': 2.8,
                'time_factor': 1.0,  # Reference speed
                'memory_factor': 1.0,  # Reference memory usage
                'strengths': ['speed', 'balanced_performance'],
                'weaknesses': ['maximum_compression']
            },
            'lzma': {
                'compression_factor': 4.2,
                'time_factor': 8.0,  # 8x slower than zstd
                'memory_factor': 2.5,
                'strengths': ['compression_ratio', 'text_data'],
                'weaknesses': ['speed', 'memory_usage']
            },
            'brotli': {
                'compression_factor': 3.8,
                'time_factor': 3.5,
                'memory_factor': 1.8,
                'strengths': ['text_compression', 'web_content'],
                'weaknesses': ['binary_data', 'memory_usage']
            },
            'paq8': {
                'compression_factor': 5.5,
                'time_factor': 50.0,  # Very slow
                'memory_factor': 8.0,  # High memory usage
                'strengths': ['maximum_compression', 'research'],
                'weaknesses': ['speed', 'memory_usage', 'practical_use']
            },
            'ac2': {
                'compression_factor': 6.0,
                'time_factor': 25.0,
                'memory_factor': 4.0,
                'strengths': ['maximum_compression', 'research'],
                'weaknesses': ['speed', 'memory_usage', 'complexity']
            }
        }
    
    def recommend_compressor(self, file_path: str, 
                           context: RecommendationContext = None) -> List[CompressorRecommendation]:
        """
        Generate ranked compressor recommendations for a file.
        
        Args:
            file_path: Path to the file to analyze
            context: Optimization context and constraints
            
        Returns:
            List of recommendations sorted by confidence score (descending)
        """
        self.logger.debug(f"Generating recommendations for: {file_path}")
        
        # Set default context if not provided
        if context is None:
            context = RecommendationContext(optimization_target='balanced')
        
        # Analyze file characteristics
        try:
            characteristics = self.file_analyzer.analyze_file(file_path)
        except Exception as e:
            self.logger.error(f"Failed to analyze file {file_path}: {e}")
            return []
        
        # Generate recommendations using multiple approaches
        heuristic_recs = self._generate_heuristic_recommendations(characteristics, context)
        historical_recs = self._generate_historical_recommendations(characteristics, context)
        domain_recs = self._generate_domain_based_recommendations(characteristics, context)
        
        # Combine and rank recommendations
        combined_recs = self._combine_recommendations([heuristic_recs, historical_recs, domain_recs])
        
        # Apply context-specific filtering and ranking
        filtered_recs = self._apply_context_filtering(combined_recs, context)
        
        # Sort by confidence score
        final_recs = sorted(filtered_recs, key=lambda r: r.confidence_score, reverse=True)
        
        self.logger.debug(f"Generated {len(final_recs)} recommendations")
        return final_recs
    
    def recommend_for_dataset(self, dataset_path: str,
                            context: RecommendationContext = None) -> Dict[str, List[CompressorRecommendation]]:
        """
        Generate recommendations for all files in a dataset directory.
        
        Args:
            dataset_path: Path to dataset directory
            context: Optimization context
            
        Returns:
            Dictionary mapping file paths to recommendation lists
        """
        dataset_path = Path(dataset_path)
        recommendations = {}
        
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                try:
                    recs = self.recommend_compressor(str(file_path), context)
                    if recs:
                        recommendations[str(file_path)] = recs
                except Exception as e:
                    self.logger.warning(f"Failed to generate recommendations for {file_path}: {e}")
        
        return recommendations
    
    def _generate_heuristic_recommendations(self, characteristics: FileCharacteristics,
                                          context: RecommendationContext) -> List[CompressorRecommendation]:
        """Generate recommendations based on heuristic rules."""
        recommendations = []
        
        # Get base recommendations from file analyzer
        suggested_compressors = characteristics.recommended_compressors
        
        for i, compressor in enumerate(suggested_compressors):
            if compressor not in self.default_expectations:
                continue
            
            expectations = self.default_expectations[compressor]
            
            # Calculate expected performance
            base_ratio = expectations['compression_factor']
            
            # Adjust based on file characteristics
            ratio_adjustment = self._calculate_ratio_adjustment(characteristics, compressor)
            expected_ratio = base_ratio * ratio_adjustment
            
            # Calculate expected time (relative to file size)
            file_size_mb = characteristics.file_size / (1024 * 1024)
            base_time = file_size_mb * 0.1  # Base: 0.1s per MB
            expected_time = base_time * expectations['time_factor']
            
            # Calculate expected memory usage
            expected_memory = max(file_size_mb * expectations['memory_factor'], 50)  # Min 50MB
            
            # Calculate confidence based on ranking and characteristics match
            base_confidence = max(0.9 - i * 0.15, 0.1)  # Decreasing confidence
            characteristic_bonus = self._calculate_characteristic_match_bonus(
                characteristics, compressor
            )
            confidence = min(base_confidence + characteristic_bonus, 1.0)
            
            # Generate reasoning
            reasoning = self._generate_heuristic_reasoning(
                characteristics, compressor, expectations
            )
            
            # Create recommendation
            rec = CompressorRecommendation(
                compressor_name=compressor,
                confidence_score=confidence,
                expected_compression_ratio=expected_ratio,
                expected_compression_time=expected_time,
                expected_memory_usage=expected_memory,
                reasoning=reasoning,
                ratio_range=(expected_ratio * 0.8, expected_ratio * 1.2),
                time_range=(expected_time * 0.7, expected_time * 1.5)
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_historical_recommendations(self, characteristics: FileCharacteristics,
                                           context: RecommendationContext) -> List[CompressorRecommendation]:
        """Generate recommendations based on historical performance data."""
        if not self.historical_data:
            return []
        
        recommendations = []
        
        # Find similar files in historical data
        similar_results = self._find_similar_historical_results(characteristics)
        
        if not similar_results:
            return []
        
        # Group by compressor and calculate statistics
        compressor_stats = {}
        for result in similar_results:
            comp = result['compressor_name']
            if comp not in compressor_stats:
                compressor_stats[comp] = {
                    'ratios': [],
                    'times': [],
                    'memory': [],
                    'results': []
                }
            
            compressor_stats[comp]['ratios'].append(result['best_compression_ratio'])
            if result['best_compression_time'] > 0:
                compressor_stats[comp]['times'].append(result['best_compression_time'])
            if result.get('best_ram_usage', 0) > 0:
                compressor_stats[comp]['memory'].append(result['best_ram_usage'])
            compressor_stats[comp]['results'].append(result)
        
        # Generate recommendations from statistics
        for compressor, stats in compressor_stats.items():
            if not stats['ratios']:
                continue
            
            # Calculate expected performance
            expected_ratio = statistics.mean(stats['ratios'])
            expected_time = statistics.mean(stats['times']) if stats['times'] else 0
            expected_memory = statistics.mean(stats['memory']) if stats['memory'] else 0
            
            # Calculate confidence based on sample size and consistency
            sample_size = len(stats['ratios'])
            consistency = 1.0 - (statistics.stdev(stats['ratios']) / expected_ratio) if sample_size > 1 else 0.5
            
            confidence = min(0.8 * (sample_size / 10) * consistency, 0.9)  # Max 0.9 for historical
            
            # Generate reasoning
            reasoning = [
                f"Based on {sample_size} similar file(s)",
                f"Average compression ratio: {expected_ratio:.2f}",
                f"Performance consistency: {consistency:.2f}"
            ]
            
            rec = CompressorRecommendation(
                compressor_name=compressor,
                confidence_score=confidence,
                expected_compression_ratio=expected_ratio,
                expected_compression_time=expected_time,
                expected_memory_usage=expected_memory,
                reasoning=reasoning,
                ratio_range=(min(stats['ratios']), max(stats['ratios'])),
                time_range=(min(stats['times']) if stats['times'] else 0,
                           max(stats['times']) if stats['times'] else 0)
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_domain_based_recommendations(self, characteristics: FileCharacteristics,
                                             context: RecommendationContext) -> List[CompressorRecommendation]:
        """Generate recommendations based on domain-specific knowledge."""
        domain = characteristics.data_domain
        data_type = characteristics.data_type
        
        # Domain-specific compressor preferences
        domain_preferences = {
            DataDomain.NATURAL_LANGUAGE: ['brotli', 'lzma', 'zstd'],
            DataDomain.SOURCE_CODE: ['lzma', 'brotli', 'zstd'], 
            DataDomain.CHEMICAL_DATA: ['lzma', 'zstd', 'brotli'],
            DataDomain.GENOMIC_DATA: ['lzma', 'paq8', 'zstd'],
            DataDomain.NUMERICAL_DATA: ['lzma', 'zstd', 'brotli'],
            DataDomain.LOG_DATA: ['lzma', 'zstd', 'brotli'],
            DataDomain.DATABASE_FILE: ['lzma', 'zstd', 'brotli'],
            DataDomain.IMAGE_DATA: ['zstd', 'brotli'],  # Usually pre-compressed
            DataDomain.CONFIGURATION: ['brotli', 'lzma', 'zstd']
        }
        
        preferred_compressors = domain_preferences.get(domain, ['zstd', 'brotli', 'lzma'])
        recommendations = []
        
        for i, compressor in enumerate(preferred_compressors):
            if compressor not in self.default_expectations:
                continue
            
            expectations = self.default_expectations[compressor]
            
            # Domain-specific performance adjustments
            domain_adjustment = self._get_domain_performance_adjustment(domain, compressor)
            expected_ratio = expectations['compression_factor'] * domain_adjustment
            
            # Base confidence on domain knowledge
            confidence = max(0.7 - i * 0.1, 0.3)
            
            # Adjust for data type compatibility
            type_bonus = self._get_data_type_compatibility_bonus(data_type, compressor)
            confidence = min(confidence + type_bonus, 0.85)  # Max 0.85 for domain-based
            
            # Calculate performance estimates
            file_size_mb = characteristics.file_size / (1024 * 1024)
            expected_time = file_size_mb * 0.1 * expectations['time_factor']
            expected_memory = max(file_size_mb * expectations['memory_factor'], 50)
            
            reasoning = [
                f"Optimized for {domain.value} data",
                f"Expected domain performance: {domain_adjustment:.2f}x baseline",
                f"Data type compatibility: {type_bonus:.2f} bonus"
            ]
            
            rec = CompressorRecommendation(
                compressor_name=compressor,
                confidence_score=confidence,
                expected_compression_ratio=expected_ratio,
                expected_compression_time=expected_time,
                expected_memory_usage=expected_memory,
                reasoning=reasoning,
                ratio_range=(expected_ratio * 0.85, expected_ratio * 1.15),
                time_range=(expected_time * 0.8, expected_time * 1.3)
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _combine_recommendations(self, recommendation_lists: List[List[CompressorRecommendation]]) -> List[CompressorRecommendation]:
        """Combine recommendations from multiple sources."""
        compressor_recs = {}
        
        # Collect all recommendations by compressor
        for rec_list in recommendation_lists:
            for rec in rec_list:
                comp = rec.compressor_name
                if comp not in compressor_recs:
                    compressor_recs[comp] = []
                compressor_recs[comp].append(rec)
        
        # Combine recommendations for each compressor
        combined = []
        for compressor, recs in compressor_recs.items():
            if not recs:
                continue
            
            # Weighted average of predictions
            total_weight = sum(r.confidence_score for r in recs)
            if total_weight == 0:
                continue
            
            weights = [r.confidence_score / total_weight for r in recs]
            
            expected_ratio = sum(r.expected_compression_ratio * w for r, w in zip(recs, weights))
            expected_time = sum(r.expected_compression_time * w for r, w in zip(recs, weights))
            expected_memory = sum(r.expected_memory_usage * w for r, w in zip(recs, weights))
            
            # Combined confidence (not simple average - boost if multiple sources agree)
            base_confidence = sum(r.confidence_score for r in recs) / len(recs)
            agreement_bonus = 0.1 * (len(recs) - 1)  # Bonus for multiple sources
            combined_confidence = min(base_confidence + agreement_bonus, 1.0)
            
            # Combine reasoning
            all_reasoning = []
            for rec in recs:
                all_reasoning.extend(rec.reasoning)
            unique_reasoning = list(dict.fromkeys(all_reasoning))  # Remove duplicates, preserve order
            
            # Combined ranges
            ratio_min = min(r.ratio_range[0] for r in recs)
            ratio_max = max(r.ratio_range[1] for r in recs)
            time_min = min(r.time_range[0] for r in recs if r.time_range[0] > 0)
            time_max = max(r.time_range[1] for r in recs if r.time_range[1] > 0)
            
            combined_rec = CompressorRecommendation(
                compressor_name=compressor,
                confidence_score=combined_confidence,
                expected_compression_ratio=expected_ratio,
                expected_compression_time=expected_time,
                expected_memory_usage=expected_memory,
                reasoning=unique_reasoning[:5],  # Limit reasoning items
                ratio_range=(ratio_min, ratio_max),
                time_range=(time_min if time_min != float('inf') else 0,
                           time_max if time_max != float('-inf') else 0)
            )
            combined.append(combined_rec)
        
        return combined
    
    def _apply_context_filtering(self, recommendations: List[CompressorRecommendation],
                               context: RecommendationContext) -> List[CompressorRecommendation]:
        """Apply context-specific filtering and ranking adjustments."""
        filtered = []
        
        for rec in recommendations:
            # Apply constraint filtering
            if context.max_time_budget and rec.expected_compression_time > context.max_time_budget:
                continue
            
            if context.memory_constraints and rec.expected_memory_usage > context.memory_constraints:
                continue
            
            # Adjust confidence based on optimization target
            target_adjustment = self._get_target_optimization_adjustment(rec, context)
            adjusted_rec = CompressorRecommendation(
                compressor_name=rec.compressor_name,
                confidence_score=min(rec.confidence_score * target_adjustment, 1.0),
                expected_compression_ratio=rec.expected_compression_ratio,
                expected_compression_time=rec.expected_compression_time,
                expected_memory_usage=rec.expected_memory_usage,
                reasoning=rec.reasoning + [f"Optimized for {context.optimization_target}"],
                ratio_range=rec.ratio_range,
                time_range=rec.time_range
            )
            filtered.append(adjusted_rec)
        
        return filtered
    
    def _calculate_ratio_adjustment(self, characteristics: FileCharacteristics, compressor: str) -> float:
        """Calculate compression ratio adjustment based on file characteristics."""
        adjustment = 1.0
        
        # Entropy-based adjustment
        entropy = characteristics.entropy
        if entropy < 4.0:  # Low entropy
            adjustment *= 1.3
        elif entropy > 7.0:  # High entropy
            adjustment *= 0.7
        
        # Text ratio adjustment
        text_ratio = characteristics.text_ratio
        if compressor in ['brotli', 'lzma'] and text_ratio > 0.8:
            adjustment *= 1.2
        elif compressor == 'zstd' and text_ratio < 0.3:  # Binary data
            adjustment *= 1.1
        
        # Repetition factor
        if characteristics.repetition_factor > 0.5:
            adjustment *= 1.2
        
        return adjustment
    
    def _calculate_characteristic_match_bonus(self, characteristics: FileCharacteristics, compressor: str) -> float:
        """Calculate bonus based on how well compressor matches file characteristics."""
        bonus = 0.0
        
        # Compressor strengths matching
        if compressor == 'brotli' and characteristics.text_ratio > 0.8:
            bonus += 0.15
        elif compressor == 'lzma' and characteristics.predicted_compressibility == 'high':
            bonus += 0.15
        elif compressor == 'zstd' and characteristics.file_size > 100 * 1024 * 1024:  # Large files
            bonus += 0.1
        
        return bonus
    
    def _find_similar_historical_results(self, characteristics: FileCharacteristics) -> List[Dict]:
        """Find similar files in historical performance data."""
        if not self.historical_data:
            return []
        
        similar_results = []
        
        # Simple similarity matching - can be improved with ML
        for result in self.historical_data.get('benchmark_results', []):
            file_chars = result.get('file_characteristics', {})
            
            # Check similarity criteria
            similarity_score = 0
            
            # Data type match
            if file_chars.get('data_type') == characteristics.data_type.value:
                similarity_score += 3
            
            # Data domain match
            if file_chars.get('data_domain') == characteristics.data_domain.value:
                similarity_score += 2
            
            # Size similarity (within order of magnitude)
            size_ratio = min(characteristics.file_size / max(file_chars.get('file_size', 1), 1),
                           max(file_chars.get('file_size', 1), 1) / characteristics.file_size)
            if size_ratio > 0.5:
                similarity_score += 1
            
            # Entropy similarity
            entropy_diff = abs(characteristics.entropy - file_chars.get('entropy', 0))
            if entropy_diff < 1.0:
                similarity_score += 1
            
            # Include if reasonably similar
            if similarity_score >= 3:
                similar_results.append(result)
        
        return similar_results
    
    def _get_domain_performance_adjustment(self, domain: DataDomain, compressor: str) -> float:
        """Get performance adjustment factor for domain-compressor combination."""
        # Cache domain performance if available
        cache_key = f"{domain.value}_{compressor}"
        if cache_key in self.domain_performance_cache:
            return self.domain_performance_cache[cache_key]
        
        # Default adjustments based on domain knowledge
        adjustments = {
            (DataDomain.NATURAL_LANGUAGE, 'brotli'): 1.2,
            (DataDomain.NATURAL_LANGUAGE, 'lzma'): 1.1,
            (DataDomain.SOURCE_CODE, 'lzma'): 1.3,
            (DataDomain.SOURCE_CODE, 'brotli'): 1.15,
            (DataDomain.CHEMICAL_DATA, 'lzma'): 1.25,
            (DataDomain.GENOMIC_DATA, 'lzma'): 1.4,
            (DataDomain.GENOMIC_DATA, 'paq8'): 1.2,
            (DataDomain.LOG_DATA, 'lzma'): 1.2,
            (DataDomain.IMAGE_DATA, 'zstd'): 1.0,  # Already compressed
        }
        
        adjustment = adjustments.get((domain, compressor), 1.0)
        self.domain_performance_cache[cache_key] = adjustment
        return adjustment
    
    def _get_data_type_compatibility_bonus(self, data_type: DataType, compressor: str) -> float:
        """Get compatibility bonus for data type and compressor combination."""
        bonuses = {
            (DataType.TEXT, 'brotli'): 0.1,
            (DataType.CODE, 'lzma'): 0.1,
            (DataType.SCIENTIFIC, 'lzma'): 0.15,
            (DataType.BINARY, 'zstd'): 0.05,
            (DataType.MULTIMEDIA, 'zstd'): 0.0,  # No bonus for already compressed
        }
        
        return bonuses.get((data_type, compressor), 0.0)
    
    def _get_target_optimization_adjustment(self, rec: CompressorRecommendation,
                                          context: RecommendationContext) -> float:
        """Get confidence adjustment based on optimization target."""
        compressor = rec.compressor_name
        target = context.optimization_target
        
        adjustments = {
            'ratio': {
                'lzma': 1.2, 'paq8': 1.3, 'ac2': 1.3,
                'brotli': 1.1, 'zstd': 0.9
            },
            'speed': {
                'zstd': 1.3, 'brotli': 1.1, 'lzma': 0.8,
                'paq8': 0.5, 'ac2': 0.6
            },
            'memory': {
                'zstd': 1.2, 'brotli': 1.0, 'lzma': 0.9,
                'paq8': 0.6, 'ac2': 0.7
            },
            'balanced': {
                'zstd': 1.2, 'brotli': 1.1, 'lzma': 1.0,
                'paq8': 0.8, 'ac2': 0.8
            }
        }
        
        return adjustments.get(target, {}).get(compressor, 1.0)
    
    def _generate_heuristic_reasoning(self, characteristics: FileCharacteristics,
                                    compressor: str, expectations: Dict) -> List[str]:
        """Generate human-readable reasoning for heuristic recommendations."""
        reasoning = []
        
        # File characteristics reasoning
        if characteristics.predicted_compressibility == 'high':
            reasoning.append("File predicted to have high compressibility")
        
        if characteristics.text_ratio > 0.8:
            reasoning.append("High text content detected")
        
        if characteristics.entropy < 5.0:
            reasoning.append("Low entropy indicates good compression potential")
        
        # Compressor strengths
        strengths = expectations.get('strengths', [])
        if 'speed' in strengths:
            reasoning.append("Fast compression and decompression")
        if 'compression_ratio' in strengths:
            reasoning.append("Excellent compression ratios")
        if 'balanced_performance' in strengths:
            reasoning.append("Good balance of speed and compression")
        
        # File size considerations
        if characteristics.file_size > 100 * 1024 * 1024 and compressor == 'zstd':
            reasoning.append("Optimized for large files")
        
        return reasoning[:4]  # Limit to 4 items
    
    def _load_knowledge_base(self):
        """Load historical performance data and domain knowledge."""
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r') as f:
                    self.historical_data = json.load(f)
                self.logger.info(f"Loaded knowledge base from {self.knowledge_base_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load knowledge base: {e}")
    
    def update_knowledge_base(self, benchmark_results: List[Dict]):
        """Update knowledge base with new benchmark results."""
        if 'benchmark_results' not in self.historical_data:
            self.historical_data['benchmark_results'] = []
        
        self.historical_data['benchmark_results'].extend(benchmark_results)
        
        # Save updated knowledge base
        try:
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(self.historical_data, f, indent=2, default=str)
            self.logger.info(f"Updated knowledge base with {len(benchmark_results)} new results")
        except Exception as e:
            self.logger.error(f"Failed to save knowledge base: {e}")
    
    def get_recommendation_summary(self, recommendations: List[CompressorRecommendation]) -> Dict[str, Any]:
        """Generate a summary of recommendations."""
        if not recommendations:
            return {'error': 'No recommendations available'}
        
        best_rec = recommendations[0]  # Highest confidence
        
        summary = {
            'primary_recommendation': {
                'compressor': best_rec.compressor_name,
                'confidence': best_rec.confidence_score,
                'expected_ratio': best_rec.expected_compression_ratio,
                'reasoning': best_rec.reasoning
            },
            'alternative_options': [
                {
                    'compressor': rec.compressor_name,
                    'confidence': rec.confidence_score,
                    'expected_ratio': rec.expected_compression_ratio
                }
                for rec in recommendations[1:3]  # Top 2 alternatives
            ],
            'total_options_evaluated': len(recommendations)
        }
        
        return summary