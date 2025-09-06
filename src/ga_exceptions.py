"""
Custom Exception Classes for Genetic Algorithm

Provides specific, meaningful exceptions for different GA failure modes
to replace generic Exception handling and fallback values.
"""

from ga_constants import MIN_FITNESS


class GAException(Exception):
    """Base exception for all genetic algorithm related errors."""
    pass


class ConfigurationError(GAException):
    """Raised when GA configuration is invalid or inconsistent."""
    pass


class CompressionError(GAException):
    """Raised when compression evaluation fails."""
    
    def __init__(self, message: str, compressor_type: str = None, 
                 parameters: dict = None, individual_name: str = None):
        super().__init__(message)
        self.compressor_type = compressor_type
        self.parameters = parameters
        self.individual_name = individual_name


class CompressionTimeoutError(CompressionError):
    """Raised when compression takes too long and times out."""
    
    def __init__(self, timeout_seconds: int, compressor_type: str = None, 
                 individual_name: str = None):
        message = f"Compression timed out after {timeout_seconds}s"
        super().__init__(message, compressor_type=compressor_type, 
                        individual_name=individual_name)
        self.timeout_seconds = timeout_seconds


class InvalidFitnessError(GAException):
    """Raised when fitness evaluation returns invalid results."""
    
    def __init__(self, fitness_value, individual_name: str = None, 
                 expected_range: tuple = None):
        message = f"Invalid fitness value: {fitness_value}"
        if expected_range:
            message += f" (expected: {expected_range[0]} to {expected_range[1]})"
        super().__init__(message)
        self.fitness_value = fitness_value
        self.individual_name = individual_name
        self.expected_range = expected_range


class PopulationError(GAException):
    """Raised when population operations fail."""
    pass


class SelectionError(GAException):
    """Raised when selection operations fail."""
    
    def __init__(self, message: str, population_size: int = None, 
                 selection_type: str = None):
        super().__init__(message)
        self.population_size = population_size
        self.selection_type = selection_type


class CrossoverError(GAException):
    """Raised when crossover operations fail."""
    
    def __init__(self, message: str, parent1_name: str = None, 
                 parent2_name: str = None):
        super().__init__(message)
        self.parent1_name = parent1_name
        self.parent2_name = parent2_name


class MutationError(GAException):
    """Raised when mutation operations fail."""
    
    def __init__(self, message: str, individual_name: str = None, 
                 mutation_rate: float = None):
        super().__init__(message)
        self.individual_name = individual_name
        self.mutation_rate = mutation_rate


class ParameterEncodingError(GAException):
    """Raised when parameter encoding/decoding fails."""
    
    def __init__(self, message: str, parameters: dict = None, 
                 encoding_type: str = None):
        super().__init__(message)
        self.parameters = parameters
        self.encoding_type = encoding_type


class ConvergenceError(GAException):
    """Raised when convergence detection fails."""
    pass


class ReportingError(GAException):
    """Raised when result reporting/saving fails."""
    
    def __init__(self, message: str, output_dir: str = None, 
                 file_type: str = None):
        super().__init__(message)
        self.output_dir = output_dir
        self.file_type = file_type


class CacheError(GAException):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_type: str = None, 
                 operation: str = None):
        super().__init__(message)
        self.cache_type = cache_type
        self.operation = operation


class ParallelProcessingError(GAException):
    """Raised when parallel processing fails."""
    
    def __init__(self, message: str, worker_count: int = None, 
                 failed_tasks: int = None):
        super().__init__(message)
        self.worker_count = worker_count
        self.failed_tasks = failed_tasks


# Error severity levels for logging
class ErrorSeverity:
    """Constants for error severity levels."""
    CRITICAL = "CRITICAL"  # GA cannot continue
    ERROR = "ERROR"       # Operation failed but GA can continue
    WARNING = "WARNING"   # Potential issue, using fallback
    INFO = "INFO"         # Normal operation info


def handle_compression_error(e: Exception, compressor_type: str, 
                            individual_name: str, use_fallback: bool = True) -> float:
    """
    Standardized compression error handler.
    
    Args:
        e: Original exception
        compressor_type: Type of compressor that failed
        individual_name: Name of individual being evaluated
        use_fallback: Whether to return fallback fitness or re-raise
        
    Returns:
        Fallback fitness value or raises CompressionError
    """
    if "timeout" in str(e).lower() or "timed out" in str(e).lower():
        error = CompressionTimeoutError(30, compressor_type, individual_name)
    else:
        error = CompressionError(
            f"Compression failed: {str(e)}", 
            compressor_type=compressor_type,
            individual_name=individual_name
        )
    
    if use_fallback:
        # Log error and return fallback value
        import logging
        logging.error(f"Compression error for {individual_name}: {error}")
        return MIN_FITNESS  # min_fitness fallback
    else:
        raise error


def validate_fitness(fitness: float, individual_name: str = None, 
                    min_expected: float = 0.0, max_expected: float = float('inf')) -> float:
    """
    Validate fitness value and raise appropriate exception if invalid.
    
    Args:
        fitness: Fitness value to validate
        individual_name: Name of individual for error context
        min_expected: Minimum expected fitness value
        max_expected: Maximum expected fitness value
        
    Returns:
        Validated fitness value
        
    Raises:
        InvalidFitnessError: If fitness is invalid
    """
    if fitness is None:
        raise InvalidFitnessError(
            "Fitness is None", 
            individual_name=individual_name,
            expected_range=(min_expected, max_expected)
        )
    
    if not isinstance(fitness, (int, float)):
        raise InvalidFitnessError(
            f"Fitness must be numeric, got {type(fitness).__name__}", 
            individual_name=individual_name
        )
    
    if fitness < min_expected or fitness > max_expected:
        raise InvalidFitnessError(
            fitness, 
            individual_name=individual_name,
            expected_range=(min_expected, max_expected)
        )
    
    return fitness