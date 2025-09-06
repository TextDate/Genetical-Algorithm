"""
Configuration Management for Genetic Algorithm

Validates and organizes user-provided CLI parameters into a clean structure.
Replaces the 12+ parameter constructor with a single config object.
"""

import os
from dataclasses import dataclass
from typing import Any
from ga_constants import GAConstants


@dataclass
class GAConfig:
    """
    Configuration container that validates and organizes user-provided parameters.
    
    Takes CLI arguments and organizes them into a validated structure,
    keeping all user control while improving code organization.
    """
    
    # Core GA Parameters (user-provided via CLI)
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    max_threads: int
    output_dir: str
    
    # Advanced GA Parameters (with sensible internal defaults)
    offspring_ratio: float = 0.9
    elite_ratio: float = 0.1
    tournament_size: int = 3
    min_fitness: float = GAConstants.MIN_FITNESS
    convergence_generations: int = 20
    convergence_threshold: float = 0.001
    
    # Algorithm Optimization Features
    enable_adaptive_tuning: bool = True
    enable_smart_initialization: bool = True
    enable_convergence_acceleration: bool = True
    enable_performance_monitoring: bool = True
    enable_dynamic_thread_scaling: bool = True
    diversity_strategy: str = "hybrid"  # "random", "structured", "hybrid"
    
    def __post_init__(self):
        """Validate user parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate critical user parameters to catch errors early."""
        errors = []
        
        # Core parameter validation
        if self.population_size < 4:
            errors.append(f"Population size ({self.population_size}) must be at least 4")
        if self.population_size % 2 != 0:
            errors.append(f"Population size ({self.population_size}) must be even for crossover")
        if self.generations < 1:
            errors.append(f"Generations ({self.generations}) must be positive")
            
        # Rate validation
        if not 0.0 <= self.mutation_rate <= 1.0:
            errors.append(f"Mutation rate ({self.mutation_rate}) must be between 0.0 and 1.0")
        if not 0.0 <= self.crossover_rate <= 1.0:
            errors.append(f"Crossover rate ({self.crossover_rate}) must be between 0.0 and 1.0")
        
        # Performance validation
        if self.max_threads < 1:
            errors.append(f"Max threads ({self.max_threads}) must be positive")
        if self.max_threads > os.cpu_count() * 4:  # Allow some flexibility
            print(f"Warning: max_threads ({self.max_threads}) is high for {os.cpu_count()} CPU cores")
            
        # Tournament validation
        if self.tournament_size > self.population_size:
            errors.append(f"Tournament size ({self.tournament_size}) cannot exceed population size ({self.population_size})")
        if self.tournament_size < 2:
            errors.append(f"Tournament size ({self.tournament_size}) must be at least 2")
            
        # Directory validation
        if not self.output_dir or not self.output_dir.strip():
            errors.append("Output directory cannot be empty")
            
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  • {error}" for error in errors))
    
    @classmethod
    def from_args(cls, args) -> 'GAConfig':
        """
        Create configuration from parsed CLI arguments.
        
        Args:
            args: argparse.Namespace from CLI parsing
            
        Returns:
            Validated GAConfig instance
        """
        return cls(
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            max_threads=args.max_threads,
            output_dir=args.output_dir
        )
    
    @property
    def num_offspring(self) -> int:
        """Calculate number of offspring based on population size and ratio."""
        return int(self.population_size * self.offspring_ratio)
    
    @property
    def num_elites(self) -> int:
        """Calculate number of elites based on population size and ratio.""" 
        return int(self.population_size * self.elite_ratio)
    
    def summary(self) -> str:
        """Generate human-readable configuration summary."""
        return f"""GA Configuration:
  Population: {self.population_size} (offspring: {self.num_offspring}, elites: {self.num_elites})
  Generations: {self.generations}
  Rates: mutation={self.mutation_rate:.3f}, crossover={self.crossover_rate:.3f}
  Threads: {self.max_threads}
  Output: {self.output_dir}"""
    
    def __str__(self) -> str:
        return f"GAConfig(pop={self.population_size}, gen={self.generations}, threads={self.max_threads})"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'max_threads': self.max_threads,
            'output_dir': self.output_dir,
            'offspring_ratio': self.offspring_ratio,
            'elite_ratio': self.elite_ratio,
            'tournament_size': self.tournament_size,
            'min_fitness': self.min_fitness,
            'convergence_generations': self.convergence_generations,
            'convergence_threshold': self.convergence_threshold,
            'enable_adaptive_tuning': self.enable_adaptive_tuning,
            'enable_smart_initialization': self.enable_smart_initialization,
            'enable_convergence_acceleration': self.enable_convergence_acceleration,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_dynamic_thread_scaling': self.enable_dynamic_thread_scaling,
            'diversity_strategy': self.diversity_strategy
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GAConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'GAConfig':
        """Create a new config with updated values."""
        current_config = self.to_dict()
        current_config.update(kwargs)
        return self.from_dict(current_config)