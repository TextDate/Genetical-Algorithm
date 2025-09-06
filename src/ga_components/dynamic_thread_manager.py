"""
Dynamic Thread Scaling Manager

Automatically adjusts thread count based on:
- Population size and workload
- System resource availability (CPU, memory)
- Current system load
- Task complexity and duration
- Performance history and optimization
"""

import os
import psutil
import threading
import time
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ga_logging import get_logger
from ga_constants import MemoryConstants, bytes_to_gb


@dataclass
class SystemResources:
    """Current system resource state."""
    cpu_count: int
    cpu_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_percent: float
    load_average: float
    
    @classmethod
    def capture(cls) -> 'SystemResources':
        """Capture current system resources."""
        cpu_count = os.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_total_gb = bytes_to_gb(memory.total)
        memory_available_gb = bytes_to_gb(memory.available)
        memory_percent = memory.percent
        
        # Load average (1-minute)
        try:
            load_average = os.getloadavg()[0] if hasattr(os, 'getloadavg') else cpu_percent / 100.0
        except (AttributeError, OSError):
            load_average = cpu_percent / 100.0
            
        return cls(
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            memory_percent=memory_percent,
            load_average=load_average
        )


@dataclass
class WorkloadCharacteristics:
    """Characteristics of the current workload."""
    population_size: int
    task_complexity: str  # 'light', 'medium', 'heavy'
    estimated_task_duration: float  # seconds per task
    memory_per_task_mb: float
    io_intensive: bool
    

@dataclass
class PerformanceMetrics:
    """Performance tracking for thread scaling decisions."""
    thread_count: int
    tasks_completed: int
    total_time: float
    throughput: float  # tasks per second
    efficiency: float  # throughput per thread
    memory_peak_gb: float
    cpu_utilization: float


class DynamicThreadManager:
    """
    Intelligent thread manager that automatically scales thread count
    based on workload and system resources.
    """
    
    def __init__(self, base_max_threads: int = None):
        """
        Initialize dynamic thread manager.
        
        Args:
            base_max_threads: Maximum thread limit (default: CPU count * 2)
        """
        self.logger = get_logger("ThreadManager")
        
        # Thread limits
        self.system_cpu_count = os.cpu_count()
        self.base_max_threads = base_max_threads or (self.system_cpu_count * 2)
        self.min_threads = 1
        self.max_threads = min(self.base_max_threads, self.system_cpu_count * 4)
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.max_history_size = 20
        
        # Current state
        self.current_threads = self.system_cpu_count
        self.last_scaling_time = 0
        self.scaling_cooldown = 30  # seconds
        
        # Adaptive parameters
        self.memory_threshold_gb = 0.8 * bytes_to_gb(psutil.virtual_memory().total)
        self.cpu_utilization_target = 0.85
        self.load_average_threshold = self.system_cpu_count * 1.5
        
        self.logger.info("Dynamic thread manager initialized",
                        system_cpus=self.system_cpu_count,
                        max_threads=self.max_threads,
                        initial_threads=self.current_threads)
    
    def calculate_optimal_threads(self, population_size: int, 
                                 task_complexity: str = 'medium',
                                 estimated_duration: float = 5.0) -> int:
        """
        Calculate optimal thread count for current workload and system state.
        
        Args:
            population_size: Number of tasks to process
            task_complexity: 'light', 'medium', or 'heavy'
            estimated_duration: Expected duration per task in seconds
            
        Returns:
            Optimal thread count
        """
        resources = SystemResources.capture()
        
        # Base calculation from workload
        workload_threads = self._calculate_workload_threads(
            population_size, task_complexity, estimated_duration
        )
        
        # Resource-based constraints
        resource_threads = self._calculate_resource_threads(resources)
        
        # Performance history optimization
        performance_threads = self._calculate_performance_threads()
        
        # Take the minimum to avoid overloading
        optimal_threads = min(workload_threads, resource_threads, performance_threads)
        
        # Apply hard limits
        optimal_threads = max(self.min_threads, min(optimal_threads, self.max_threads))
        
        self.logger.debug("Thread calculation breakdown",
                         workload_threads=workload_threads,
                         resource_threads=resource_threads,
                         performance_threads=performance_threads,
                         optimal_threads=optimal_threads,
                         population_size=population_size,
                         complexity=task_complexity)
        
        return optimal_threads
    
    def _calculate_workload_threads(self, population_size: int, 
                                  task_complexity: str, 
                                  estimated_duration: float) -> int:
        """Calculate threads needed based on workload characteristics."""
        
        # Complexity multipliers
        complexity_factors = {
            'light': 1.5,    # Can handle more tasks per thread
            'medium': 1.0,   # Standard ratio
            'heavy': 0.7     # Fewer tasks per thread
        }
        
        factor = complexity_factors.get(task_complexity, 1.0)
        
        # Base threads from population size
        if population_size <= 4:
            base_threads = population_size  # Don't over-parallelize small populations
        elif population_size <= 20:
            base_threads = min(self.system_cpu_count, population_size // 2)
        else:
            base_threads = min(self.system_cpu_count * 2, population_size // 4)
        
        # Adjust for task duration (longer tasks benefit more from parallelization)
        if estimated_duration > 10:
            duration_factor = 1.3
        elif estimated_duration > 30:
            duration_factor = 1.5
        else:
            duration_factor = 1.0
            
        workload_threads = int(base_threads * factor * duration_factor)
        
        return max(1, workload_threads)
    
    def _calculate_resource_threads(self, resources: SystemResources) -> int:
        """Calculate thread limit based on current system resources."""
        
        # CPU-based limit
        cpu_factor = max(0.3, (100 - resources.cpu_percent) / 100)  # More available CPU = more threads
        cpu_threads = int(self.system_cpu_count * cpu_factor * 2)
        
        # Memory-based limit (assume ~100MB per thread for compression tasks)
        memory_per_thread_gb = 0.1
        memory_threads = int(resources.memory_available_gb / memory_per_thread_gb)
        
        # Load average limit
        load_factor = max(0.5, 2.0 - (resources.load_average / self.system_cpu_count))
        load_threads = int(self.system_cpu_count * load_factor)
        
        # Take the most restrictive limit
        resource_threads = min(cpu_threads, memory_threads, load_threads)
        
        self.logger.debug("Resource-based thread calculation",
                         cpu_percent=resources.cpu_percent,
                         cpu_threads=cpu_threads,
                         memory_available_gb=resources.memory_available_gb,
                         memory_threads=memory_threads,
                         load_average=resources.load_average,
                         load_threads=load_threads,
                         final_resource_threads=resource_threads)
        
        return max(1, resource_threads)
    
    def _calculate_performance_threads(self) -> int:
        """Calculate thread count based on performance history."""
        if len(self.performance_history) < 3:
            return self.max_threads  # Not enough history
        
        # Find the thread count with best efficiency in recent history
        recent_metrics = self.performance_history[-10:]  # Last 10 runs
        
        # Group by thread count and calculate average efficiency
        thread_efficiency = {}
        for metric in recent_metrics:
            if metric.thread_count not in thread_efficiency:
                thread_efficiency[metric.thread_count] = []
            thread_efficiency[metric.thread_count].append(metric.efficiency)
        
        # Calculate average efficiency for each thread count
        avg_efficiency = {}
        for thread_count, efficiencies in thread_efficiency.items():
            avg_efficiency[thread_count] = statistics.mean(efficiencies)
        
        # Find optimal thread count (highest efficiency)
        if avg_efficiency:
            optimal = max(avg_efficiency.items(), key=lambda x: x[1])[0]
            self.logger.debug("Performance-based optimization", 
                            thread_efficiencies=avg_efficiency,
                            optimal_threads=optimal)
            return optimal
            
        return self.max_threads
    
    def get_threads_for_workload(self, population_size: int, 
                                generation: int = 0,
                                compressor_type: str = 'medium') -> int:
        """
        Main interface: get optimal thread count for current workload.
        
        Args:
            population_size: Number of individuals to evaluate
            generation: Current generation (for complexity estimation)
            compressor_type: Type of compressor for complexity estimation
            
        Returns:
            Optimal thread count for this workload
        """
        # Estimate task complexity based on compressor
        complexity_map = {
            'zstd': 'light',
            'brotli': 'medium', 
            'lzma': 'heavy',
            'paq8': 'heavy'
        }
        
        task_complexity = complexity_map.get(compressor_type.lower(), 'medium')
        
        # Estimate duration based on complexity and generation
        duration_map = {
            'light': 2.0,
            'medium': 5.0,
            'heavy': 15.0
        }
        base_duration = duration_map[task_complexity]
        
        # Later generations might be more complex due to parameter optimization
        generation_factor = 1.0 + (generation * 0.02)  # 2% increase per generation
        estimated_duration = base_duration * generation_factor
        
        optimal_threads = self.calculate_optimal_threads(
            population_size, task_complexity, estimated_duration
        )
        
        # Check scaling cooldown
        current_time = time.time()
        if (current_time - self.last_scaling_time < self.scaling_cooldown and
            abs(optimal_threads - self.current_threads) <= 2):
            # Small change within cooldown period - keep current
            optimal_threads = self.current_threads
        else:
            self.last_scaling_time = current_time
            self.current_threads = optimal_threads
        
        self.logger.info("Thread scaling decision",
                        population_size=population_size,
                        generation=generation,
                        compressor_type=compressor_type,
                        task_complexity=task_complexity,
                        estimated_duration=f"{estimated_duration:.1f}s",
                        optimal_threads=optimal_threads)
        
        return optimal_threads
    
    def record_performance(self, thread_count: int, tasks_completed: int,
                          total_time: float, peak_memory_gb: float,
                          cpu_utilization: float = None) -> None:
        """
        Record performance metrics for future optimization.
        
        Args:
            thread_count: Number of threads used
            tasks_completed: Number of tasks completed
            total_time: Total execution time in seconds
            peak_memory_gb: Peak memory usage in GB
            cpu_utilization: Average CPU utilization (optional)
        """
        if total_time <= 0 or tasks_completed <= 0:
            return
            
        throughput = tasks_completed / total_time
        efficiency = throughput / thread_count if thread_count > 0 else 0
        
        if cpu_utilization is None:
            cpu_utilization = psutil.cpu_percent(interval=0.1)
        
        metric = PerformanceMetrics(
            thread_count=thread_count,
            tasks_completed=tasks_completed,
            total_time=total_time,
            throughput=throughput,
            efficiency=efficiency,
            memory_peak_gb=peak_memory_gb,
            cpu_utilization=cpu_utilization
        )
        
        self.performance_history.append(metric)
        
        # Keep history size manageable
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
        
        self.logger.debug("Performance recorded",
                         thread_count=thread_count,
                         throughput=f"{throughput:.2f} tasks/s",
                         efficiency=f"{efficiency:.2f} tasks/s/thread",
                         memory_peak_gb=f"{peak_memory_gb:.2f}GB")
    
    def get_system_status(self) -> Dict[str, any]:
        """Get current system status and recommendations."""
        resources = SystemResources.capture()
        
        # Resource utilization assessment
        memory_pressure = resources.memory_percent > 80
        cpu_pressure = resources.cpu_percent > 85
        load_pressure = resources.load_average > (self.system_cpu_count * 1.2)
        
        # Performance trend analysis
        trend = "stable"
        if len(self.performance_history) >= 5:
            recent_efficiency = [m.efficiency for m in self.performance_history[-5:]]
            if statistics.mean(recent_efficiency[-3:]) > statistics.mean(recent_efficiency[:2]):
                trend = "improving"
            elif statistics.mean(recent_efficiency[-3:]) < statistics.mean(recent_efficiency[:2]):
                trend = "declining"
        
        recommendations = []
        if memory_pressure:
            recommendations.append("Consider reducing thread count - high memory usage")
        if cpu_pressure:
            recommendations.append("System CPU saturated - threads may not help")  
        if load_pressure:
            recommendations.append("High system load - reduce parallelization")
        
        return {
            'system_resources': resources,
            'current_threads': self.current_threads,
            'max_threads': self.max_threads,
            'memory_pressure': memory_pressure,
            'cpu_pressure': cpu_pressure,
            'load_pressure': load_pressure,
            'performance_trend': trend,
            'recommendations': recommendations,
            'performance_history_size': len(self.performance_history)
        }


# Global thread manager instance
_thread_manager: Optional[DynamicThreadManager] = None


def get_thread_manager(base_max_threads: int = None) -> DynamicThreadManager:
    """Get or create the global thread manager instance."""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = DynamicThreadManager(base_max_threads)
    return _thread_manager


def get_optimal_threads(population_size: int, generation: int = 0, 
                       compressor_type: str = 'medium') -> int:
    """
    Convenience function to get optimal thread count.
    
    Args:
        population_size: Number of tasks to process
        generation: Current generation number  
        compressor_type: Type of compressor for complexity estimation
        
    Returns:
        Optimal thread count
    """
    manager = get_thread_manager()
    return manager.get_threads_for_workload(population_size, generation, compressor_type)