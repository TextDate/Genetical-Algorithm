"""
Centralized Logging System for Genetic Algorithm

Replaces scattered print statements with structured logging.
Provides consistent formatting, log levels, and file output.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
from pathlib import Path


class GAFormatter(logging.Formatter):
    """Custom formatter for GA logging with color support and structured output."""
    
    # Color codes for console output
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors: bool = True, include_timestamp: bool = True):
        self.use_colors = use_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        self.include_timestamp = include_timestamp
        
        # Format string
        if include_timestamp:
            fmt = '[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s'
            datefmt = '%H:%M:%S'
        else:
            fmt = '%(levelname)-8s | %(name)s | %(message)s'
            datefmt = None
            
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        # Apply colors for console output
        if self.use_colors:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


class GALogger:
    """
    Centralized logger for genetic algorithm with console and file output.
    
    Manages log levels, file rotation, and provides GA-specific logging methods.
    """
    
    def __init__(self, name: str = "GA", level: str = "INFO", 
                 log_to_file: bool = True, output_dir: str = "logs",
                 console_colors: bool = True):
        """
        Initialize GA logger.
        
        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            output_dir: Directory for log files
            console_colors: Whether to use colors in console output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = GAFormatter(use_colors=console_colors, include_timestamp=False)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if requested
        if log_to_file:
            self._setup_file_logging(output_dir)
    
    def _setup_file_logging(self, output_dir: str):
        """Setup file logging with rotation."""
        # Create logs directory
        log_dir = Path(output_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ga_run_{timestamp}.log"
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        file_formatter = GAFormatter(use_colors=False, include_timestamp=True)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Store file path for reference
        self.log_file = str(log_file)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log error message with optional exception details."""
        formatted_msg = self._format_message(message, **kwargs)
        if exception:
            formatted_msg += f" | Exception: {type(exception).__name__}: {str(exception)}"
        self.logger.error(formatted_msg)
    
    def critical(self, message: str, exception: Exception = None, **kwargs):
        """Log critical message with optional exception details."""
        formatted_msg = self._format_message(message, **kwargs)
        if exception:
            formatted_msg += f" | Exception: {type(exception).__name__}: {str(exception)}"
        self.logger.critical(formatted_msg)
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with optional context parameters."""
        if kwargs:
            context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} | {context}"
        return message
    
    # GA-specific logging methods
    def log_generation_start(self, generation: int, population_size: int):
        """Log generation start."""
        self.info(f"Starting generation {generation}", 
                 population_size=population_size)
    
    def log_generation_complete(self, generation: int, best_fitness: float, 
                              time_taken: float, peak_memory: float):
        """Log generation completion."""
        self.info(f"Generation {generation} complete", 
                 best_fitness=f"{best_fitness:.4f}",
                 time_taken=f"{time_taken:.2f}s",
                 peak_memory=f"{peak_memory:.2f}GB")
    
    def log_evaluation_error(self, individual_name: str, compressor_type: str, 
                           error: Exception, using_fallback: bool = True):
        """Log evaluation error."""
        action = "using fallback fitness" if using_fallback else "skipping individual"
        self.error(f"Evaluation failed for {individual_name}", 
                  compressor=compressor_type, 
                  action=action,
                  exception=error)
    
    def log_compression_timeout(self, individual_name: str, timeout_seconds: int):
        """Log compression timeout."""
        self.warning(f"Compression timeout for {individual_name}", 
                    timeout=f"{timeout_seconds}s")
    
    def log_convergence(self, generation: int, reason: str):
        """Log convergence detection."""
        self.info(f"Convergence detected at generation {generation}", 
                 reason=reason)
    
    def log_cache_stats(self, hits: int, misses: int, hit_rate: float):
        """Log cache statistics."""
        self.info("Cache performance", 
                 hits=hits, 
                 misses=misses, 
                 hit_rate=f"{hit_rate:.1%}")
    
    def log_config_summary(self, config):
        """Log configuration summary."""
        self.info("GA Configuration loaded", 
                 population=config.population_size,
                 generations=config.generations,
                 threads=config.max_threads,
                 mutation_rate=config.mutation_rate,
                 crossover_rate=config.crossover_rate)
    
    def log_parallel_processing(self, worker_count: int, task_count: int, 
                              time_taken: float):
        """Log parallel processing performance."""
        self.debug("Parallel processing complete", 
                  workers=worker_count,
                  tasks=task_count,
                  time_taken=f"{time_taken:.2f}s",
                  tasks_per_second=f"{task_count/time_taken:.1f}")


# Global logger instance
_global_logger: Optional[GALogger] = None


def get_logger(name: str = "GA") -> GALogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = GALogger(name)
    return _global_logger


def setup_logging(level: str = "INFO", log_to_file: bool = True, 
                  output_dir: str = "logs", console_colors: bool = True) -> GALogger:
    """
    Setup global logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        output_dir: Directory for log files
        console_colors: Whether to use colors in console output
        
    Returns:
        Configured GALogger instance
    """
    global _global_logger
    _global_logger = GALogger(
        level=level,
        log_to_file=log_to_file,
        output_dir=output_dir,
        console_colors=console_colors
    )
    return _global_logger


def log_exception(exception: Exception, context: str = "", **kwargs):
    """Log exception with context using global logger."""
    logger = get_logger()
    logger.error(f"Exception in {context}", exception=exception, **kwargs)