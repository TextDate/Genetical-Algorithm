"""
Evaluation Module

Handles fitness evaluation with parallel processing, memory monitoring,
and performance optimization for genetic algorithms.

Features:
- Parallel fitness evaluation using ProcessPoolExecutor
- Memory usage tracking and monitoring
- Error handling and timeout management
- Evaluation statistics and performance metrics
"""

import concurrent.futures
import threading
import time
import sys
from functools import lru_cache
from typing import List, Tuple, Any, Dict, Optional
import psutil
from tqdm import tqdm
from ga_exceptions import (
    CompressionError, CompressionTimeoutError, InvalidFitnessError,
    ParallelProcessingError, handle_compression_error, validate_fitness
)
from ga_logging import get_logger
from ga_components.dynamic_thread_manager import get_thread_manager
from ga_components.compressor_registry import create_compressor_config
from ga_components.multi_objective_evaluator import MultiObjectiveEvaluator, ObjectiveWeights


def evaluate_fitness_worker(args: Tuple) -> float:
    """
    Worker function for ProcessPoolExecutor that recreates compressor locally.
    Avoids serialization issues by reconstructing objects in worker process.
    
    Args:
        args: Tuple of (individual, compressor_config, decoded_individual)
        
    Returns:
        Fitness value
    """
    try:
        individual, compressor_config, decoded_individual = args
        gene_code, individual_name = individual
        
        # Reconstruct compressor from config
        compressor_type = compressor_config['type']
        compressor_args = compressor_config['args']
        
        # Import and create the appropriate compressor
        if compressor_type == 'ZstdCompressor':
            from Compressors.zstd_compressor import ZstdCompressor
            compressor = ZstdCompressor(**compressor_args)
        elif compressor_type == 'LzmaCompressor':
            from Compressors.lzma_compressor import LzmaCompressor  
            compressor = LzmaCompressor(**compressor_args)
        elif compressor_type == 'BrotliCompressor':
            from Compressors.brotli_compressor import BrotliCompressor
            compressor = BrotliCompressor(**compressor_args)
        elif compressor_type == 'PAQ8Compressor':
            from Compressors.paq8_compressor import PAQ8Compressor
            compressor = PAQ8Compressor(**compressor_args)
        elif compressor_type == 'AC2Compressor':
            from Compressors.ac2_compressor import AC2Compressor
            compressor = AC2Compressor(**compressor_args)
        else:
            raise CompressionError(
                f"Unknown compressor type: {compressor_type}",
                compressor_type=compressor_type,
                individual_name=individual_name
            )
        
        # File-based multiprocess cache is automatically used
        
        # Evaluate fitness and get timing
        result = compressor.evaluate(decoded_individual, individual_name)
        
        # Handle tuple return (fitness, compression_time) or just fitness for backward compatibility
        if isinstance(result, tuple):
            fitness, compression_time = result
        else:
            fitness = result
            compression_time = 0.0
        
        # Validate fitness result with proper error handling
        validated_fitness = validate_fitness(fitness, individual_name, min_expected=0.0)
        return validated_fitness, compression_time
            
    except Exception as e:
        # Use standardized error handling
        fallback_fitness = handle_compression_error(
            e, compressor_config.get('type', 'unknown'), individual_name
        )
        return fallback_fitness, 0.0  # Return tuple (fitness, time)


class EvaluationEngine:
    """
    High-performance evaluation engine for genetic algorithms.
    
    Provides parallel fitness evaluation with memory monitoring,
    error handling, and performance optimization.
    """
    
    def __init__(self, compressor: Any, population_manager: Any, max_threads: int = 8, 
                 min_fitness: float = 0.1, enable_dynamic_scaling: bool = True, config: Any = None):
        """
        Initialize evaluation engine.
        
        Args:
            compressor: Compressor instance for fitness evaluation
            population_manager: Population manager for decoding
            max_threads: Maximum number of parallel threads (used as base limit for dynamic scaling)
            min_fitness: Minimum fitness for failed evaluations
            enable_dynamic_scaling: Whether to use dynamic thread scaling
            config: GAConfig instance for multi-objective evaluation settings
        """
        self.compressor = compressor
        self.population_manager = population_manager
        self.base_max_threads = max_threads
        self.max_threads = max_threads  # This will be dynamically updated
        self.min_fitness = min_fitness
        self.enable_dynamic_scaling = enable_dynamic_scaling
        self.config = config
        
        # Initialize thread manager if dynamic scaling enabled
        if self.enable_dynamic_scaling:
            self.thread_manager = get_thread_manager(self.base_max_threads)
        else:
            self.thread_manager = None
        self.logger = get_logger("EvaluationEngine")
        
        # Initialize multi-objective evaluator if configured
        self.multi_objective_evaluator = None
        if config and hasattr(config, 'enable_multi_objective') and config.enable_multi_objective:
            weights = ObjectiveWeights(
                fitness_weight=config.fitness_weight,
                time_weight=config.time_weight,
                additional_weights=config.additional_objectives or {}
            )
            self.multi_objective_evaluator = MultiObjectiveEvaluator(
                weights=weights,
                normalize_time=config.normalize_time,
                enable_time_penalty=config.enable_time_penalty,
                time_penalty_threshold=config.time_penalty_threshold
            )
            self.logger.info("Multi-objective evaluation enabled",
                           fitness_weight=config.fitness_weight,
                           time_weight=config.time_weight)
        
        # Create serializable compressor config for ProcessPoolExecutor using registry
        self.compressor_config = create_compressor_config(compressor)
        
        # Statistics
        self.stats = {
            'evaluations_performed': 0,
            'parallel_batches': 0,
            'evaluation_errors': 0,
            'total_evaluation_time': 0.0,
            'peak_memory_usage': 0.0
        }
    
    # _create_compressor_config method removed - now using registry pattern
    # See ga_components.compressor_registry for the new implementation
    
    def evaluate_fitness(self, individual: Tuple[Tuple[str, ...], str]) -> float:
        """
        Evaluate fitness of a single individual with proper error handling.
        
        Args:
            individual: Individual to evaluate (gene_code, individual_name)
            
        Returns:
            Fitness value
            
        Raises:
            CompressionError: If compression fails and fallback is disabled
            InvalidFitnessError: If fitness validation fails
        """
        try:
            # Simple decoding without local cache (GA level cache will handle this)
            gene_code, individual_name = individual
            decoded_individual = self.population_manager.decode_individual(individual)
            
            # Evaluate with proper error context and get timing
            compressor_type = type(self.compressor).__name__
            result = self.compressor.evaluate(decoded_individual, individual_name)
            
            # Handle tuple return (fitness, compression_time) or just fitness for backward compatibility
            if isinstance(result, tuple):
                fitness, compression_time = result
            else:
                fitness = result
                compression_time = 0.0
            
            # Validate fitness with proper error handling
            validated_fitness = validate_fitness(fitness, individual_name, min_expected=0.0)
            
            self.stats['evaluations_performed'] += 1
            return validated_fitness, compression_time
            
        except Exception as e:
            self.stats['evaluation_errors'] += 1
            compressor_type = type(self.compressor).__name__
            
            # Use standardized error handling with logging
            self.logger.log_evaluation_error(individual[1], compressor_type, e, using_fallback=True)
            fallback_fitness = handle_compression_error(e, compressor_type, individual[1])
            return fallback_fitness, 0.0  # Return tuple (fitness, time)
    
    def evaluate_population_parallel(self, population: List[Tuple[Tuple[str, ...], str]], 
                                   generation: int = 0) -> Tuple[List[float], List[float], float]:
        """
        Evaluate entire population in parallel with memory monitoring and dynamic scaling.
        
        Args:
            population: Population to evaluate
            generation: Current generation number for dynamic scaling
            
        Returns:
            Tuple of (fitness_results, compression_times, peak_memory_gb)
            - fitness_results: Combined multi-objective scores if enabled, otherwise raw compression ratios
            - compression_times: Individual compression times for reporting
            - peak_memory_gb: Peak memory usage during evaluation
        """
        start_time = time.time()
        peak_memory = 0.0
        memory_monitor_active = threading.Event()
        memory_monitor_active.set()
        
        def track_memory():
            """Background thread to monitor memory usage across all processes."""
            nonlocal peak_memory
            while memory_monitor_active.is_set():
                try:
                    # Get main process
                    main_process = psutil.Process()
                    total_memory = main_process.memory_info().rss
                    
                    # Add memory from all child processes (worker processes)
                    try:
                        children = main_process.children(recursive=True)
                        for child in children:
                            try:
                                total_memory += child.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                                # Child process may have terminated
                                continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Main process issues, but continue with just main process memory
                        pass
                    
                    # Convert to GB and update peak
                    memory_gb = total_memory / 1024**3
                    peak_memory = max(peak_memory, memory_gb)
                    time.sleep(0.1)  # Check every 100ms for better accuracy during intensive operations
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        
        # Start memory monitoring thread
        memory_thread = threading.Thread(target=track_memory, daemon=True)
        memory_thread.start()
        
        try:
            # Prepare arguments for ProcessPoolExecutor
            evaluation_args = []
            for individual in population:
                # Decode individual here (in main process) to avoid serialization issues
                decoded_individual = self.population_manager.decode_individual(individual)
                evaluation_args.append((individual, self.compressor_config, decoded_individual))
            
            # Dynamic thread scaling with safety cap
            if self.enable_dynamic_scaling and self.thread_manager:
                compressor_type = type(self.compressor).__name__.lower().replace('compressor', '')
                optimal_threads = self.thread_manager.get_threads_for_workload(
                    len(population), generation, compressor_type
                )
                # Cap threads to reasonable maximum and respect base_max_threads from config
                self.max_threads = min(optimal_threads, self.base_max_threads, 16)
                
                self.logger.info("Dynamic thread scaling applied", 
                               base_max_threads=self.base_max_threads,
                               optimal_threads=optimal_threads,
                               population_size=len(population),
                               generation=generation,
                               compressor_type=compressor_type)
            
            # Log parallel processing start
            self.logger.debug(f"Starting parallel evaluation", 
                            workers=self.max_threads, 
                            tasks=len(evaluation_args))
            
            # Parallel evaluation with ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
                evaluation_results = list(
                    tqdm(executor.map(evaluate_fitness_worker, evaluation_args), 
                         total=len(population),
                         desc=f"Evaluating Population ({self.max_threads} workers)")
                )
                
                # Separate fitness values and compression times
                fitness_results = []
                compression_times = []
                for result in evaluation_results:
                    if isinstance(result, tuple):
                        fitness, comp_time = result
                        fitness_results.append(fitness)
                        compression_times.append(comp_time)
                    else:
                        # Backward compatibility for old results
                        fitness_results.append(result)
                        compression_times.append(0.0)
            
            # Log successful parallel processing
            evaluation_time = time.time() - start_time
            self.logger.log_parallel_processing(
                self.max_threads, len(evaluation_args), evaluation_time
            )
            
            # Record performance metrics for dynamic scaling optimization
            if self.enable_dynamic_scaling and self.thread_manager:
                # Get current CPU utilization (approximate)
                cpu_utilization = psutil.cpu_percent(interval=0.1)
                peak_memory_gb = peak_memory
                
                self.thread_manager.record_performance(
                    thread_count=self.max_threads,
                    tasks_completed=len(evaluation_args),
                    total_time=evaluation_time,
                    peak_memory_gb=peak_memory_gb,
                    cpu_utilization=cpu_utilization
                )
                
        except Exception as e:
            self.logger.error("Parallel evaluation failed, falling back to sequential", 
                            exception=e, workers=self.max_threads)
            
            # Fallback to sequential evaluation
            fitness_results = []
            compression_times = []
            for individual in tqdm(population, desc="Evaluating Population (Sequential)"):
                try:
                    result = self.evaluate_fitness(individual)
                    if isinstance(result, tuple):
                        fitness, comp_time = result
                        fitness_results.append(fitness)
                        compression_times.append(comp_time)
                    else:
                        fitness_results.append(result)
                        compression_times.append(0.0)
                except Exception as eval_error:
                    # Log and use fallback for individual failures
                    self.logger.log_evaluation_error(
                        individual[1], type(self.compressor).__name__, eval_error
                    )
                    fitness_results.append(self.min_fitness)
                    compression_times.append(0.0)
        finally:
            # Stop memory monitoring
            memory_monitor_active.clear()
            if memory_thread.is_alive():
                memory_thread.join(timeout=1)
        
        # Apply multi-objective evaluation if enabled
        if self.multi_objective_evaluator:
            evaluation_metrics = self.multi_objective_evaluator.evaluate_population(
                fitness_results, compression_times
            )
            # Replace fitness_results with combined multi-objective scores
            fitness_results = self.multi_objective_evaluator.get_combined_scores(evaluation_metrics)
            
            self.logger.debug("Multi-objective evaluation applied",
                           avg_combined_score=sum(fitness_results) / len(fitness_results) if fitness_results else 0.0,
                           generation=generation)
        
        # Update statistics
        evaluation_time = time.time() - start_time
        self.stats['parallel_batches'] += 1
        self.stats['total_evaluation_time'] += evaluation_time
        self.stats['peak_memory_usage'] = max(self.stats['peak_memory_usage'], peak_memory)
        
        return fitness_results, compression_times, peak_memory
    
    def evaluate_batch_sequential(self, population: List[Tuple[Tuple[str, ...], str]]) -> List[float]:
        """
        Evaluate population sequentially (fallback method).
        
        Args:
            population: Population to evaluate
            
        Returns:
            List of fitness values
        """
        start_time = time.time()
        fitness_results = []
        
        for individual in tqdm(population, desc="Evaluating Population"):
            fitness = self.evaluate_fitness(individual)
            fitness_results.append(fitness)
        
        # Update statistics
        evaluation_time = time.time() - start_time
        self.stats['total_evaluation_time'] += evaluation_time
        
        return fitness_results
    
    def evaluate_with_timeout(self, individual: Tuple[Tuple[str, ...], str], 
                             timeout_seconds: int = 30) -> Optional[float]:
        """
        Evaluate individual with timeout protection.
        
        Args:
            individual: Individual to evaluate
            timeout_seconds: Maximum evaluation time
            
        Returns:
            Fitness value or None if timeout
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.evaluate_fitness, individual)
                fitness = future.result(timeout=timeout_seconds)
                return fitness
        except concurrent.futures.TimeoutError:
            self.logger.warning("Evaluation timeout", 
                              timeout_seconds=timeout_seconds,
                              individual=individual[1])
            return None
        except Exception as e:
            self.logger.error("Error in timeout evaluation", 
                             individual=individual[1], exception=e)
            return None
    
    def benchmark_evaluation(self, sample_population: List[Tuple[Tuple[str, ...], str]], 
                            num_runs: int = 3) -> Dict[str, float]:
        """
        Benchmark evaluation performance.
        
        Args:
            sample_population: Sample population for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        sequential_times = []
        parallel_times = []
        
        # No cache to clear
        
        for run in range(num_runs):
            # Sequential benchmark
            start_time = time.time()
            self.evaluate_batch_sequential(sample_population)
            sequential_times.append(time.time() - start_time)
            
            # Parallel benchmark (now sequential)
            start_time = time.time()
            self.evaluate_population_parallel(sample_population)
            parallel_times.append(time.time() - start_time)
        
        return {
            'sequential_avg_time': sum(sequential_times) / len(sequential_times),
            'parallel_avg_time': sum(parallel_times) / len(parallel_times),
            'speedup_ratio': sum(sequential_times) / sum(parallel_times),
            'population_size': len(sample_population),
            'max_threads': self.max_threads
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get evaluation cache statistics (disabled - no cache)."""
        return {
            'hits': 0,
            'misses': 0,
            'hit_rate': 0.0,
            'current_size': 0,
            'max_size': 0
        }
    
    def clear_cache(self):
        """Clear evaluation cache (disabled - no cache)."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        stats = self.stats.copy()
        
        # Add cache statistics
        cache_info = self.get_cache_info()
        stats.update({
            'cache_hits': cache_info['hits'],
            'cache_misses': cache_info['misses'],
            'cache_hit_rate': cache_info['hit_rate'],
            'cache_size': cache_info['current_size']
        })
        
        # Add multi-objective statistics if enabled
        if self.multi_objective_evaluator:
            multi_obj_stats = self.multi_objective_evaluator.get_statistics()
            stats['multi_objective'] = multi_obj_stats
        
        # Add derived metrics
        if self.stats['evaluations_performed'] > 0:
            stats['avg_evaluation_time'] = self.stats['total_evaluation_time'] / self.stats['evaluations_performed']
            stats['error_rate'] = self.stats['evaluation_errors'] / self.stats['evaluations_performed']
        else:
            stats['avg_evaluation_time'] = 0.0
            stats['error_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset all evaluation statistics."""
        self.stats = {
            'evaluations_performed': 0,
            'parallel_batches': 0,
            'evaluation_errors': 0,
            'total_evaluation_time': 0.0,
            'peak_memory_usage': 0.0
        }
        
        # Reset multi-objective statistics if enabled
        if self.multi_objective_evaluator:
            self.multi_objective_evaluator.reset_statistics()
    
    def optimize_thread_count(self, sample_population: List[Tuple[Tuple[str, ...], str]]) -> int:
        """
        Automatically determine optimal thread count for this system.
        
        Args:
            sample_population: Sample population for testing
            
        Returns:
            Optimal thread count
        """
        import multiprocessing
        
        max_threads = multiprocessing.cpu_count()
        thread_counts = [1, 2, 4, max_threads // 2, max_threads]
        thread_counts = sorted(set(t for t in thread_counts if t > 0))
        
        best_threads = 1
        best_time = float('inf')
        
        original_max_threads = self.max_threads
        
        for threads in thread_counts:
            self.max_threads = threads
            # No cache to clear
            
            start_time = time.time()
            self.evaluate_population_parallel(sample_population[:min(10, len(sample_population))])
            evaluation_time = time.time() - start_time
            
            if evaluation_time < best_time:
                best_time = evaluation_time
                best_threads = threads
        
        # Restore original setting
        self.max_threads = original_max_threads
        
        return best_threads