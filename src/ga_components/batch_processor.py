"""
Batch Processing System for Multi-Dataset Operations

Provides automated batch processing capabilities for running genetic algorithm
optimization across multiple datasets, files, and compressor combinations.
"""

import os
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import threading
from queue import Queue
import psutil

from ga_logging import get_logger, setup_logging
from ga_config import GAConfig
from ga_components.file_analyzer import FileAnalyzer, FileCharacteristics
from ga_components.compressor_recommender import CompressorRecommender, RecommendationContext
from ga_components.multi_domain_benchmarker import MultiDomainBenchmarker, MultiDomainBenchmarkConfig


@dataclass
class BatchJob:
    """Individual batch processing job."""
    job_id: str
    file_path: str
    compressor: str
    param_ranges: Dict[str, Any]
    ga_config: GAConfig
    output_dir: str
    priority: int = 1  # Higher number = higher priority
    
    # Optional constraints
    max_runtime: Optional[float] = None
    memory_limit_mb: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'file_path': self.file_path,
            'compressor': self.compressor,
            'param_ranges': self.param_ranges,
            'ga_config': asdict(self.ga_config),
            'output_dir': self.output_dir,
            'priority': self.priority,
            'max_runtime': self.max_runtime,
            'memory_limit_mb': self.memory_limit_mb
        }


@dataclass
class BatchJobResult:
    """Result from a completed batch job."""
    job: BatchJob
    success: bool
    start_time: float
    end_time: float
    best_fitness: Optional[float] = None
    best_parameters: Optional[Dict] = None
    error_message: Optional[str] = None
    
    # Performance metrics
    peak_memory_mb: Optional[float] = None
    cpu_usage_avg: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job': self.job.to_dict(),
            'success': self.success,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'runtime': self.end_time - self.start_time,
            'best_fitness': self.best_fitness,
            'best_parameters': self.best_parameters,
            'error_message': self.error_message,
            'peak_memory_mb': self.peak_memory_mb,
            'cpu_usage_avg': self.cpu_usage_avg
        }


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    # Resource limits
    max_concurrent_jobs: int = 4
    max_memory_per_job_mb: int = 2048
    max_total_memory_mb: int = 8192
    cpu_cores_per_job: int = 2
    
    # Processing options
    enable_smart_scheduling: bool = True
    enable_resource_monitoring: bool = True
    retry_failed_jobs: bool = True
    max_retries: int = 2
    
    # Output configuration
    save_individual_results: bool = True
    save_progress_reports: bool = True
    compress_output: bool = False
    
    # Job selection
    auto_recommend_compressors: bool = True
    use_file_characteristics_filtering: bool = True
    skip_large_files_mb: Optional[int] = None
    skip_small_files_kb: Optional[int] = None


class ResourceMonitor:
    """System resource monitoring for batch processing."""
    
    def __init__(self):
        self.logger = get_logger("ResourceMonitor")
        self.monitoring = False
        self._stop_event = threading.Event()
        
    def start_monitoring(self, callback=None):
        """Start resource monitoring."""
        self.monitoring = True
        self._stop_event.clear()
        
        def monitor_loop():
            while not self._stop_event.is_set():
                try:
                    memory_usage = psutil.virtual_memory().percent
                    cpu_usage = psutil.cpu_percent(interval=1)
                    
                    if callback:
                        callback(memory_usage, cpu_usage)
                        
                    if memory_usage > 90:
                        self.logger.warning(f"High memory usage: {memory_usage:.1f}%")
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        self._stop_event.set()


class BatchProcessor:
    """
    Comprehensive batch processing system for multi-dataset genetic algorithm optimization.
    
    Handles job scheduling, resource management, progress tracking, and result aggregation
    for large-scale compression optimization experiments.
    """
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.logger = get_logger("BatchProcessor")
        self.file_analyzer = FileAnalyzer()
        self.compressor_recommender = CompressorRecommender()
        self.resource_monitor = ResourceMonitor()
        
        # Job management
        self.job_queue: Queue = Queue()
        self.completed_jobs: List[BatchJobResult] = []
        self.failed_jobs: List[BatchJobResult] = []
        self.active_jobs: Dict[str, BatchJob] = {}
        
        # Threading
        self._processing_lock = threading.Lock()
        self._job_counter = 0
        
    def create_batch_jobs(self, dataset_configs: List[Dict]) -> List[BatchJob]:
        """
        Create batch jobs from dataset configurations.
        
        Args:
            dataset_configs: List of configuration dictionaries containing:
                - dataset_path: Path to dataset directory
                - compressors: List of compressors to test (optional)
                - file_patterns: File patterns to include (optional)
                - ga_config: GA configuration overrides (optional)
        
        Returns:
            List of BatchJob objects
        """
        self.logger.info("Creating batch jobs from dataset configurations")
        all_jobs = []
        
        for config in dataset_configs:
            dataset_path = config['dataset_path']
            
            if not os.path.exists(dataset_path):
                self.logger.warning(f"Dataset path not found: {dataset_path}")
                continue
            
            # Discover files in dataset
            files = self._discover_dataset_files(config)
            self.logger.info(f"Discovered {len(files)} files in {dataset_path}")
            
            # Determine compressors to use
            compressors = config.get('compressors', ['zstd', 'lzma', 'brotli'])
            
            # Load parameter configurations
            param_configs = self._load_parameter_configs()
            
            # Create jobs for each file-compressor combination
            for file_path, characteristics in files:
                # Apply file filtering
                if not self._should_process_file(characteristics):
                    continue
                
                # Get compressor recommendations if enabled
                if self.config.auto_recommend_compressors:
                    recommendations = self.compressor_recommender.recommend_compressor(file_path)
                    recommended_compressors = [r.compressor_name for r in recommendations[:2]]  # Top 2
                    compressors_to_use = [c for c in compressors if c in recommended_compressors] or compressors[:1]
                else:
                    compressors_to_use = compressors
                
                # Create jobs for selected compressors
                for compressor in compressors_to_use:
                    if compressor not in param_configs:
                        self.logger.warning(f"No parameters found for compressor: {compressor}")
                        continue
                    
                    job = self._create_individual_job(
                        file_path, compressor, param_configs[compressor], config, characteristics
                    )
                    all_jobs.append(job)
        
        self.logger.info(f"Created {len(all_jobs)} batch jobs")
        return all_jobs
    
    def process_batch(self, jobs: List[BatchJob], output_dir: str) -> Dict[str, Any]:
        """
        Process a batch of jobs with intelligent scheduling and resource management.
        
        Args:
            jobs: List of BatchJob objects to process
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing batch processing results and statistics
        """
        self.logger.info(f"Starting batch processing of {len(jobs)} jobs")
        start_time = time.time()
        
        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Setup logging for batch processing
        batch_logger = setup_logging(
            level="INFO",
            log_to_file=True,
            output_dir=output_dir,
            console_colors=True
        )
        
        # Start resource monitoring if enabled
        if self.config.enable_resource_monitoring:
            self.resource_monitor.start_monitoring(self._resource_callback)
        
        try:
            # Sort jobs by priority and resource requirements
            if self.config.enable_smart_scheduling:
                jobs = self._optimize_job_schedule(jobs)
            
            # Add jobs to queue
            for job in jobs:
                self.job_queue.put(job)
            
            # Process jobs with controlled concurrency
            self._process_jobs_concurrent()
            
            # Generate final report
            results = self._generate_batch_report(start_time)
            
            # Save results
            self._save_batch_results(results, output_dir)
            
            return results
            
        finally:
            # Cleanup
            if self.config.enable_resource_monitoring:
                self.resource_monitor.stop_monitoring()
    
    def process_multi_domain_benchmark(self, dataset_dirs: List[str], 
                                     compressors: List[str],
                                     output_dir: str) -> Dict[str, Any]:
        """
        Process multi-domain benchmark across multiple dataset directories.
        
        Args:
            dataset_dirs: List of dataset directory paths
            compressors: List of compressors to benchmark
            output_dir: Output directory for results
            
        Returns:
            Comprehensive benchmark results
        """
        self.logger.info(f"Starting multi-domain benchmark across {len(dataset_dirs)} datasets")
        
        # Create multi-domain benchmark configuration
        benchmark_config = MultiDomainBenchmarkConfig(
            dataset_dir="",  # Will be set per dataset
            output_dir=os.path.join(output_dir, "multi_domain_results"),
            compressors=compressors,
            max_concurrent_benchmarks=self.config.max_concurrent_jobs,
            ga_config=GAConfig(
                population_size=20,  # Reduced for batch processing
                generations=15,
                max_threads=self.config.cpu_cores_per_job
            )
        )
        
        all_results = []
        
        # Process each dataset
        for dataset_dir in dataset_dirs:
            if not os.path.exists(dataset_dir):
                self.logger.warning(f"Dataset directory not found: {dataset_dir}")
                continue
            
            self.logger.info(f"Processing dataset: {dataset_dir}")
            benchmark_config.dataset_dir = dataset_dir
            
            # Create benchmarker and run
            benchmarker = MultiDomainBenchmarker(benchmark_config)
            try:
                results = benchmarker.run_benchmark()
                results['dataset_path'] = dataset_dir
                all_results.append(results)
            except Exception as e:
                self.logger.error(f"Failed to benchmark dataset {dataset_dir}: {e}")
        
        # Aggregate results across all datasets
        aggregated_results = self._aggregate_multi_domain_results(all_results)
        
        # Save aggregated results
        output_file = os.path.join(output_dir, "aggregated_multi_domain_results.json")
        with open(output_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        
        self.logger.info(f"Multi-domain benchmark completed. Results saved to {output_dir}")
        return aggregated_results
    
    def _discover_dataset_files(self, config: Dict) -> List[Tuple[str, FileCharacteristics]]:
        """Discover and analyze files in a dataset."""
        dataset_path = Path(config['dataset_path'])
        file_patterns = config.get('file_patterns', ['*'])
        
        files_with_characteristics = []
        
        for pattern in file_patterns:
            for file_path in dataset_path.rglob(pattern):
                if file_path.is_file():
                    try:
                        characteristics = self.file_analyzer.analyze_file(str(file_path))
                        files_with_characteristics.append((str(file_path), characteristics))
                    except Exception as e:
                        self.logger.debug(f"Failed to analyze {file_path}: {e}")
        
        return files_with_characteristics
    
    def _should_process_file(self, characteristics: FileCharacteristics) -> bool:
        """Determine if a file should be processed based on filtering criteria."""
        file_size = characteristics.file_size
        
        # Size-based filtering
        if self.config.skip_large_files_mb:
            if file_size > self.config.skip_large_files_mb * 1024 * 1024:
                return False
        
        if self.config.skip_small_files_kb:
            if file_size < self.config.skip_small_files_kb * 1024:
                return False
        
        # Characteristics-based filtering
        if self.config.use_file_characteristics_filtering:
            # Skip already compressed files
            if characteristics.predicted_compressibility == "low" and "compress" in characteristics.file_path.lower():
                return False
        
        return True
    
    def _load_parameter_configs(self) -> Dict[str, Dict]:
        """Load parameter configurations for all compressors."""
        param_file_paths = [
            "config/params.json",
            "../config/params.json",
            "params.json"
        ]
        
        for path in param_file_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        
        # Return minimal default if no config found
        self.logger.warning("No parameter configuration file found, using defaults")
        return {
            'zstd': {'level': [1, 3, 6, 9, 12]},
            'lzma': {'preset': [0, 3, 6, 9]},
            'brotli': {'quality': [4, 6, 8, 11]}
        }
    
    def _create_individual_job(self, file_path: str, compressor: str, 
                             param_ranges: Dict, dataset_config: Dict,
                             characteristics: FileCharacteristics) -> BatchJob:
        """Create an individual batch job."""
        self._job_counter += 1
        job_id = f"job_{self._job_counter:04d}_{compressor}_{Path(file_path).stem}"
        
        # Create GA config with appropriate settings for batch processing
        ga_config_overrides = dataset_config.get('ga_config', {})
        
        ga_config = GAConfig(
            population_size=ga_config_overrides.get('population_size', 20),
            generations=ga_config_overrides.get('generations', 15),
            max_threads=min(self.config.cpu_cores_per_job, ga_config_overrides.get('max_threads', 2)),
            output_dir=os.path.join(self.output_dir, "individual_jobs", job_id),
            mutation_rate=ga_config_overrides.get('mutation_rate', 0.01),
            crossover_rate=ga_config_overrides.get('crossover_rate', 0.8)
        )
        
        # Determine priority based on file characteristics
        priority = self._calculate_job_priority(characteristics, compressor)
        
        # Estimate resource requirements
        estimated_memory = max(characteristics.file_size / (1024 * 1024) * 2, 500)  # 2x file size, min 500MB
        memory_limit = min(estimated_memory, self.config.max_memory_per_job_mb)
        
        # Estimate runtime
        max_runtime = self._estimate_job_runtime(characteristics, compressor, ga_config)
        
        return BatchJob(
            job_id=job_id,
            file_path=file_path,
            compressor=compressor,
            param_ranges=param_ranges,
            ga_config=ga_config,
            output_dir=ga_config.output_dir,
            priority=priority,
            max_runtime=max_runtime,
            memory_limit_mb=int(memory_limit)
        )
    
    def _calculate_job_priority(self, characteristics: FileCharacteristics, compressor: str) -> int:
        """Calculate job priority based on characteristics."""
        priority = 5  # Base priority
        
        # Prioritize files with high compression potential
        if characteristics.predicted_compressibility == "high":
            priority += 2
        elif characteristics.predicted_compressibility == "low":
            priority -= 1
        
        # Prioritize smaller files (process faster)
        if characteristics.file_size < 10 * 1024 * 1024:  # < 10MB
            priority += 1
        elif characteristics.file_size > 100 * 1024 * 1024:  # > 100MB
            priority -= 1
        
        # Prioritize faster compressors
        if compressor == 'zstd':
            priority += 1
        elif compressor in ['paq8', 'ac2']:
            priority -= 2
        
        return max(1, priority)  # Minimum priority 1
    
    def _estimate_job_runtime(self, characteristics: FileCharacteristics, 
                            compressor: str, ga_config: GAConfig) -> float:
        """Estimate job runtime in seconds."""
        file_size_mb = characteristics.file_size / (1024 * 1024)
        
        # Base time per compression (seconds per MB)
        base_times = {
            'zstd': 0.1,
            'brotli': 0.3,
            'lzma': 1.0,
            'paq8': 10.0,
            'ac2': 5.0
        }
        
        base_time = base_times.get(compressor, 1.0)
        compression_time = file_size_mb * base_time
        
        # Account for GA iterations
        total_evaluations = ga_config.population_size * ga_config.generations
        estimated_runtime = compression_time * total_evaluations / ga_config.max_threads
        
        # Add overhead (30%)
        return estimated_runtime * 1.3
    
    def _optimize_job_schedule(self, jobs: List[BatchJob]) -> List[BatchJob]:
        """Optimize job scheduling order for better resource utilization."""
        # Sort by priority (high to low), then by estimated runtime (low to high)
        return sorted(jobs, key=lambda j: (-j.priority, j.max_runtime or 0))
    
    def _process_jobs_concurrent(self):
        """Process jobs with controlled concurrency."""
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_jobs) as executor:
            futures = []
            
            # Submit initial batch of jobs
            for _ in range(min(self.config.max_concurrent_jobs, self.job_queue.qsize())):
                if not self.job_queue.empty():
                    job = self.job_queue.get()
                    future = executor.submit(self._process_single_job, job)
                    futures.append((future, job))
            
            # Process completed jobs and submit new ones
            while futures:
                # Wait for at least one job to complete
                completed_futures = []
                for future, job in futures:
                    if future.done():
                        completed_futures.append((future, job))
                
                # Process completed jobs
                for future, job in completed_futures:
                    futures.remove((future, job))
                    
                    try:
                        result = future.result()
                        if result.success:
                            self.completed_jobs.append(result)
                        else:
                            self.failed_jobs.append(result)
                            
                            # Retry if configured
                            if self.config.retry_failed_jobs and job.priority > -self.config.max_retries:
                                job.priority = -abs(job.priority) - 1  # Mark as retry
                                self.job_queue.put(job)
                        
                        self.logger.info(f"Job {job.job_id} completed: {'SUCCESS' if result.success else 'FAILED'}")
                        
                    except Exception as e:
                        self.logger.error(f"Job {job.job_id} failed with exception: {e}")
                
                # Submit new jobs if available
                while len(futures) < self.config.max_concurrent_jobs and not self.job_queue.empty():
                    job = self.job_queue.get()
                    future = executor.submit(self._process_single_job, job)
                    futures.append((future, job))
                
                # Brief pause to avoid busy waiting
                if not completed_futures:
                    time.sleep(1)
    
    def _process_single_job(self, job: BatchJob) -> BatchJobResult:
        """Process a single batch job."""
        self.logger.debug(f"Starting job {job.job_id}")
        start_time = time.time()
        
        # Monitor process resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        cpu_measurements = []
        
        try:
            # Import and run GA (imports here to avoid multiprocessing issues)
            from genetic_algorithm import GeneticAlgorithm
            from Compressors.zstd_compressor import ZstdCompressor
            from Compressors.lzma_compressor import LzmaCompressor
            from Compressors.brotli_compressor import BrotliCompressor
            from Compressors.paq8_compressor import PAQ8Compressor
            from Compressors.ac2_compressor import AC2Compressor
            
            # Create compressor instance
            temp_dir = os.path.join(job.output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            if job.compressor == 'zstd':
                compressor = ZstdCompressor(job.file_path, temp=temp_dir)
            elif job.compressor == 'lzma':
                compressor = LzmaCompressor(job.file_path, temp=temp_dir)
            elif job.compressor == 'brotli':
                compressor = BrotliCompressor(job.file_path, temp=temp_dir)
            elif job.compressor == 'paq8':
                compressor = PAQ8Compressor(job.file_path, temp=temp_dir)
            elif job.compressor == 'ac2':
                # Skip AC2 for batch processing (requires reference)
                return BatchJobResult(
                    job=job,
                    success=False,
                    start_time=start_time,
                    end_time=time.time(),
                    error_message="AC2 compressor not supported in batch processing"
                )
            else:
                return BatchJobResult(
                    job=job,
                    success=False,
                    start_time=start_time,
                    end_time=time.time(),
                    error_message=f"Unknown compressor: {job.compressor}"
                )
            
            # Create and run GA
            os.makedirs(job.ga_config.output_dir, exist_ok=True)
            ga = GeneticAlgorithm(job.param_ranges, compressor, job.ga_config)
            
            # Run with timeout if specified
            if job.max_runtime:
                # Note: This is a simplified timeout - in production you'd want more robust timeout handling
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Job exceeded maximum runtime")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(job.max_runtime))
            
            try:
                best_solution, best_fitness = ga.run()
                
                # Record final resource usage
                end_time = time.time()
                final_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, final_memory)
                
                return BatchJobResult(
                    job=job,
                    success=True,
                    start_time=start_time,
                    end_time=end_time,
                    best_fitness=best_fitness,
                    best_parameters=best_solution,
                    peak_memory_mb=peak_memory,
                    cpu_usage_avg=sum(cpu_measurements) / len(cpu_measurements) if cpu_measurements else 0
                )
                
            finally:
                if job.max_runtime:
                    signal.alarm(0)  # Cancel timeout
                
                # Cleanup
                try:
                    compressor.erase_temp_files()
                except:
                    pass
            
        except Exception as e:
            return BatchJobResult(
                job=job,
                success=False,
                start_time=start_time,
                end_time=time.time(),
                error_message=str(e),
                peak_memory_mb=peak_memory
            )
    
    def _resource_callback(self, memory_percent: float, cpu_percent: float):
        """Callback for resource monitoring."""
        if memory_percent > 85:
            self.logger.warning(f"System memory usage high: {memory_percent:.1f}%")
        
        if len(self.active_jobs) > 0 and cpu_percent < 20:
            self.logger.debug(f"Low CPU usage: {cpu_percent:.1f}% with {len(self.active_jobs)} active jobs")
    
    def _generate_batch_report(self, start_time: float) -> Dict[str, Any]:
        """Generate comprehensive batch processing report."""
        end_time = time.time()
        total_runtime = end_time - start_time
        
        total_jobs = len(self.completed_jobs) + len(self.failed_jobs)
        success_rate = len(self.completed_jobs) / total_jobs if total_jobs > 0 else 0
        
        # Calculate performance statistics
        successful_results = [job for job in self.completed_jobs if job.best_fitness]
        
        if successful_results:
            avg_fitness = sum(job.best_fitness for job in successful_results) / len(successful_results)
            best_overall = max(successful_results, key=lambda j: j.best_fitness)
        else:
            avg_fitness = 0
            best_overall = None
        
        # Group results by compressor
        compressor_stats = {}
        for job in self.completed_jobs:
            comp = job.job.compressor
            if comp not in compressor_stats:
                compressor_stats[comp] = {'count': 0, 'avg_fitness': 0, 'total_runtime': 0}
            
            compressor_stats[comp]['count'] += 1
            compressor_stats[comp]['avg_fitness'] += job.best_fitness or 0
            compressor_stats[comp]['total_runtime'] += job.end_time - job.start_time
        
        # Calculate averages
        for comp_stats in compressor_stats.values():
            if comp_stats['count'] > 0:
                comp_stats['avg_fitness'] /= comp_stats['count']
                comp_stats['avg_runtime'] = comp_stats['total_runtime'] / comp_stats['count']
        
        report = {
            'batch_summary': {
                'total_jobs': total_jobs,
                'successful_jobs': len(self.completed_jobs),
                'failed_jobs': len(self.failed_jobs),
                'success_rate': success_rate,
                'total_runtime': total_runtime,
                'avg_job_runtime': total_runtime / total_jobs if total_jobs > 0 else 0
            },
            'performance_summary': {
                'average_fitness': avg_fitness,
                'best_overall_fitness': best_overall.best_fitness if best_overall else None,
                'best_overall_job': best_overall.job.job_id if best_overall else None,
                'best_overall_compressor': best_overall.job.compressor if best_overall else None
            },
            'compressor_statistics': compressor_stats,
            'detailed_results': [job.to_dict() for job in self.completed_jobs],
            'failed_jobs': [job.to_dict() for job in self.failed_jobs]
        }
        
        return report
    
    def _save_batch_results(self, results: Dict[str, Any], output_dir: str):
        """Save batch processing results."""
        # Save main results
        results_file = os.path.join(output_dir, "batch_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_file = os.path.join(output_dir, "batch_summary.csv")
        with open(csv_file, 'w', newline='') as f:
            fieldnames = [
                'job_id', 'file_name', 'compressor', 'success', 
                'best_fitness', 'runtime', 'peak_memory_mb'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for job in self.completed_jobs + self.failed_jobs:
                writer.writerow({
                    'job_id': job.job.job_id,
                    'file_name': Path(job.job.file_path).name,
                    'compressor': job.job.compressor,
                    'success': job.success,
                    'best_fitness': job.best_fitness or 0,
                    'runtime': job.end_time - job.start_time,
                    'peak_memory_mb': job.peak_memory_mb or 0
                })
        
        self.logger.info(f"Batch results saved to {output_dir}")
        self.logger.info(f"JSON Results: {results_file}")
        self.logger.info(f"CSV Summary: {csv_file}")
    
    def _aggregate_multi_domain_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple domain benchmarks."""
        if not all_results:
            return {'error': 'No results to aggregate'}
        
        # Combine all raw results
        combined_raw_results = []
        for result in all_results:
            if 'raw_results' in result:
                combined_raw_results.extend(result['raw_results'])
        
        # Aggregate compressor performance across all datasets
        overall_compressor_performance = {}
        for result in all_results:
            comp_perf = result.get('compressor_performance', {})
            for compressor, metrics in comp_perf.items():
                if compressor not in overall_compressor_performance:
                    overall_compressor_performance[compressor] = {
                        'total_benchmarks': 0,
                        'avg_compression_ratios': [],
                        'avg_runtimes': [],
                        'datasets_tested': []
                    }
                
                overall_compressor_performance[compressor]['total_benchmarks'] += metrics['benchmarks_run']
                overall_compressor_performance[compressor]['avg_compression_ratios'].append(metrics['avg_compression_ratio'])
                overall_compressor_performance[compressor]['avg_runtimes'].append(metrics['avg_total_runtime'])
                overall_compressor_performance[compressor]['datasets_tested'].append(result.get('dataset_path', 'unknown'))
        
        # Calculate final aggregated metrics
        for compressor, data in overall_compressor_performance.items():
            data['overall_avg_ratio'] = sum(data['avg_compression_ratios']) / len(data['avg_compression_ratios'])
            data['overall_avg_runtime'] = sum(data['avg_runtimes']) / len(data['avg_runtimes'])
            data['datasets_count'] = len(set(data['datasets_tested']))
        
        aggregated = {
            'summary': {
                'datasets_processed': len(all_results),
                'total_benchmarks': len(combined_raw_results),
                'compressors_evaluated': list(overall_compressor_performance.keys())
            },
            'overall_compressor_performance': overall_compressor_performance,
            'individual_dataset_results': all_results,
            'combined_raw_results': combined_raw_results
        }
        
        return aggregated