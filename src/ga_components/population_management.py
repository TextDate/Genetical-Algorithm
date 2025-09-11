"""
Population Management Module

Handles population initialization, individual creation, decoding, and various
population-level utilities for genetic algorithms.

Features:
- Random population initialization
- Individual naming and tracking  
- Binary to parameter decoding
- Population validation and repair
- Individual comparison utilities
"""

import random
from typing import Dict, List, Tuple, Any, Set


class PopulationManager:
    """
    Manages population-level operations for genetic algorithms.
    
    Handles creation, validation, and manipulation of populations
    with support for binary-encoded individuals.
    """
    
    def __init__(self, param_binary_encodings: Dict[str, Dict[str, Any]], 
                 population_size: int, compressor_nr_models: int):
        """
        Initialize population manager.
        
        Args:
            param_binary_encodings: Parameter encoding specifications
            population_size: Target population size
            compressor_nr_models: Number of models per individual
        """
        
        self.param_binary_encodings = param_binary_encodings
        self.population_size = population_size
        self.compressor_nr_models = compressor_nr_models
        
        # Individual naming
        self.individual_counter = 0
        
        # Statistics
        self.stats = {
            'individuals_created': 0,
            'populations_initialized': 0,
            'decode_operations': 0
        }
    
    def initialize_population(self) -> List[Tuple[Tuple[str, ...], str]]:
        """
        Initialize a random population with unique individuals.
        
        Returns:
            List of individuals [(gene_code, individual_name), ...]
        """
        population = []
        seen_signatures = set()
        
        
        while len(population) < self.population_size:
            # Generate individual
            individual = self._create_random_individual(1)
            signature = self._individual_to_signature(individual)
            
            # Ensure uniqueness
            if signature not in seen_signatures:
                population.append(individual)
                seen_signatures.add(signature)
            
            # Prevent infinite loops
            if len(seen_signatures) > 100 * self.population_size:
                # Fill remaining slots with slight variations
                while len(population) < self.population_size:
                    base_individual = random.choice(population)
                    varied_individual = self._create_variation(base_individual, 1)
                    population.append(varied_individual)
                break
        
        self.stats['populations_initialized'] += 1
        
        
        return population
    
    def _create_random_individual(self, generation: int) -> Tuple[Tuple[str, ...], str]:
        """
        Create a single random individual.
        
        Args:
            generation: Generation number for naming
            
        Returns:
            Individual tuple (gene_code, individual_name)
        """
        gene_code = []
        
        # Generate genes for each model
        for model_idx in range(self.compressor_nr_models):
            gene = ''
            
            # Build gene by concatenating binary representations
            expected_params = len(self.param_binary_encodings)
            expected_bits = sum(enc['bit_length'] for enc in self.param_binary_encodings.values())
            
            
            param_count = 0
            for param, encodings in self.param_binary_encodings.items():
                param_count += 1
                value_idx = random.randint(0, len(encodings['values']) - 1)
                param_bits = encodings['binary_map'][value_idx]
                gene += param_bits
                
            
            # Validate gene was created correctly
            if len(gene) != expected_bits:
                raise ValueError(f"Gene creation failed: expected {expected_bits} bits but got {len(gene)} bits. "
                               f"Processed {param_count}/{expected_params} parameters. Gene: '{gene}'. "
                               f"Available params: {list(self.param_binary_encodings.keys())}")
            
            # Validate we processed all parameters
            if param_count != expected_params:
                raise ValueError(f"Gene creation incomplete: processed {param_count}/{expected_params} parameters")
            
            
            gene_code.append(gene)
        
        individual_name = self.create_individual_name(generation)
        
        self.stats['individuals_created'] += 1
        return (tuple(gene_code), individual_name)
    
    def _create_variation(self, base_individual: Tuple[Tuple[str, ...], str], 
                         generation: int, mutation_strength: float = 0.1) -> Tuple[Tuple[str, ...], str]:
        """
        Create a variation of an existing individual.
        
        Args:
            base_individual: Individual to vary
            generation: Generation for naming
            mutation_strength: Strength of variation
            
        Returns:
            Varied individual
        """
        gene_code, _ = base_individual
        varied_gene_code = []
        
        for gene in gene_code:
            varied_gene = ""
            
            # Apply light mutation
            for bit in gene:
                if random.random() < mutation_strength:
                    varied_gene += '1' if bit == '0' else '0'
                else:
                    varied_gene += bit
            
            varied_gene_code.append(varied_gene)
        
        new_name = self.create_individual_name(generation)
        self.stats['individuals_created'] += 1
        
        return (tuple(varied_gene_code), new_name)
    
    def create_individual_name(self, generation: int) -> str:
        """
        Create a unique name for an individual.
        
        Args:
            generation: Generation number
            
        Returns:
            Unique individual name
        """
        self.individual_counter += 1
        return f"Gen{generation}_Ind{self.individual_counter}"
    
    def decode_individual(self, individual: Tuple[Tuple[str, ...], str]) -> List[Dict[str, Any]]:
        """
        Decode binary individual to parameter values.
        
        Args:
            individual: Individual to decode (gene_code, name)
            
        Returns:
            List of decoded parameter dictionaries (one per model)
        """
        gene_code, _ = individual
        decoded_individual = []
        
        for gene in gene_code:
            decoded_gene = {}
            current_pos = 0
            
            for param, encodings in self.param_binary_encodings.items():
                bit_length = encodings['bit_length']
                binary_value = gene[current_pos:current_pos + bit_length]
                
                # Convert binary to value index
                if not binary_value:
                    raise ValueError(f"Empty binary value for parameter '{param}' at position {current_pos} in gene: {gene}")
                
                value_idx = int(binary_value, 2)
                
                # Validate index and correct if needed (use deterministic correction)
                if value_idx >= len(encodings['values']):
                    value_idx = len(encodings['values']) - 1  # Use max valid index instead of random
                
                decoded_gene[param] = encodings['values'][value_idx]
                current_pos += bit_length
            
            decoded_individual.append(decoded_gene)
        
        self.stats['decode_operations'] += 1
        return decoded_individual
    
    def _individual_to_signature(self, individual: Tuple[Tuple[str, ...], str]) -> str:
        """
        Convert individual to unique signature for comparison.
        
        Args:
            individual: Individual to convert
            
        Returns:
            Unique signature string
        """
        gene_code, _ = individual
        return ''.join(gene_code)
    
    def validate_population(self, population: List[Tuple[Tuple[str, ...], str]]) -> List[Tuple[Tuple[str, ...], str]]:
        """
        Validate and repair population if needed.
        
        Args:
            population: Population to validate
            
        Returns:
            Validated population
        """
        validated_population = []
        
        for individual in population:
            if self._validate_individual(individual):
                validated_population.append(individual)
            else:
                # Repair or replace invalid individual
                repaired = self._repair_individual(individual, 1)
                validated_population.append(repaired)
        
        return validated_population
    
    def _validate_individual(self, individual: Tuple[Tuple[str, ...], str]) -> bool:
        """
        Check if individual is valid.
        
        Args:
            individual: Individual to validate
            
        Returns:
            True if valid, False otherwise
        """
        gene_code, name = individual
        
        # Check structure
        if not isinstance(gene_code, tuple) or len(gene_code) != self.compressor_nr_models:
            return False
        
        # Check gene lengths
        expected_length = sum(enc['bit_length'] for enc in self.param_binary_encodings.values())
        
        for gene in gene_code:
            if not isinstance(gene, str) or len(gene) != expected_length:
                return False
            
            # Check if gene contains only 0s and 1s
            if not all(bit in '01' for bit in gene):
                return False
        
        return True
    
    def _repair_individual(self, individual: Tuple[Tuple[str, ...], str], 
                          generation: int) -> Tuple[Tuple[str, ...], str]:
        """
        Repair an invalid individual.
        
        Args:
            individual: Individual to repair
            generation: Generation for naming
            
        Returns:
            Repaired individual
        """
        # For simplicity, create a new random individual
        # In practice, you might want more sophisticated repair
        return self._create_random_individual(generation)
    
    def get_population_diversity(self, population: List[Tuple[Tuple[str, ...], str]]) -> float:
        """
        Calculate population diversity based on unique signatures.
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity ratio (0.0 to 1.0)
        """
        if not population:
            return 0.0
        
        signatures = set()
        for individual in population:
            signatures.add(self._individual_to_signature(individual))
        
        return len(signatures) / len(population)
    
    def find_duplicates(self, population: List[Tuple[Tuple[str, ...], str]]) -> Dict[str, List[int]]:
        """
        Find duplicate individuals in population.
        
        Args:
            population: Population to analyze
            
        Returns:
            Dictionary mapping signatures to lists of indices
        """
        signature_map = {}
        
        for i, individual in enumerate(population):
            signature = self._individual_to_signature(individual)
            
            if signature not in signature_map:
                signature_map[signature] = []
            signature_map[signature].append(i)
        
        # Return only signatures with duplicates
        duplicates = {sig: indices for sig, indices in signature_map.items() 
                     if len(indices) > 1}
        
        return duplicates
    
    def get_statistics(self) -> Dict[str, int]:
        """Get population management statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'individuals_created': 0,
            'populations_initialized': 0,
            'decode_operations': 0
        }
    
    def clone_individual(self, individual: Tuple[Tuple[str, ...], str], 
                        generation: int) -> Tuple[Tuple[str, ...], str]:
        """
        Create an exact clone of an individual with new name.
        
        Args:
            individual: Individual to clone
            generation: Generation for naming
            
        Returns:
            Cloned individual with new name
        """
        gene_code, _ = individual
        new_name = self.create_individual_name(generation)
        self.stats['individuals_created'] += 1
        
        return (gene_code, new_name)