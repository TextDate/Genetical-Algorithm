"""
Parameter Encoding Module

Handles binary encoding and decoding of genetic algorithm parameters.
Provides utilities for parameter space analysis and encoding optimization.

Features:
- Binary parameter encoding with variable bit lengths
- Parameter space size calculations
- Encoding validation and optimization
- Support for various parameter types (numeric, categorical)
"""

import math
from typing import Dict, List, Any, Union, Tuple


class ParameterEncoder:
    """
    Handles encoding and decoding of genetic algorithm parameters.
    
    Converts parameter values to binary representations for genetic operations
    and provides utilities for parameter space analysis.
    """
    
    def __init__(self, param_values: Dict[str, List[Any]]):
        """
        Initialize parameter encoder.
        
        Args:
            param_values: Dictionary mapping parameter names to their possible values
        """
        self.param_values = param_values
        self.param_binary_encodings = self._encode_parameters()
        self.total_parameter_combinations = self._calculate_total_parameter_space()
        
        # Statistics
        self.stats = {
            'parameters_encoded': len(param_values),
            'total_bits_required': self._calculate_total_bits(),
            'encoding_efficiency': self._calculate_encoding_efficiency()
        }
    
    def _encode_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Encode parameters to binary representation.
        
        Returns:
            Dictionary containing encoding information for each parameter
        """
        param_encodings = {}
        
        for param, values in self.param_values.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"Parameter '{param}' must have a non-empty list of values")
            
            # Calculate required bits for this parameter
            num_bits = len(bin(len(values) - 1)[2:])  # Number of bits required
            
            # Create binary representations
            binary_representations = {}
            for i in range(len(values)):
                binary_representations[i] = format(i, f'0{num_bits}b')
            
            param_encodings[param] = {
                'values': values,
                'binary_map': binary_representations,
                'bit_length': num_bits,
                'value_count': len(values),
                'param_type': self._infer_parameter_type(values)
            }
        
        return param_encodings
    
    def _infer_parameter_type(self, values: List[Any]) -> str:
        """
        Infer the type of parameter from its values.
        
        Args:
            values: List of parameter values
            
        Returns:
            Parameter type string
        """
        if not values:
            return 'unknown'
        
        first_value = values[0]
        
        if isinstance(first_value, (int, float)):
            # Check if all values are numeric
            if all(isinstance(v, (int, float)) for v in values):
                # Check if it's a range of integers
                if all(isinstance(v, int) for v in values):
                    sorted_values = sorted(values)
                    if sorted_values == list(range(sorted_values[0], sorted_values[-1] + 1)):
                        return 'integer_range'
                    else:
                        return 'integer_discrete'
                else:
                    return 'numeric'
        
        elif isinstance(first_value, str):
            if all(isinstance(v, str) for v in values):
                return 'categorical'
        
        elif isinstance(first_value, bool):
            if all(isinstance(v, bool) for v in values):
                return 'boolean'
        
        return 'mixed'
    
    def _calculate_total_parameter_space(self) -> int:
        """
        Calculate total number of possible parameter combinations.
        
        Returns:
            Total parameter space size (capped at 10M for stability)
        """
        total_combinations = 1
        
        for param_name, param_values in self.param_values.items():
            total_combinations *= len(param_values)
            
            # Prevent numerical overflow
            if total_combinations > 10_000_000:
                return 10_000_000
        
        return total_combinations
    
    def _calculate_total_bits(self) -> int:
        """Calculate total bits required for encoding all parameters."""
        return sum(enc['bit_length'] for enc in self.param_binary_encodings.values())
    
    def _calculate_encoding_efficiency(self) -> float:
        """
        Calculate encoding efficiency (how well we use the bit space).
        
        Returns:
            Efficiency ratio between 0.0 and 1.0
        """
        total_bits = self._calculate_total_bits()
        if total_bits == 0:
            return 0.0
        
        # Calculate actual combinations vs possible combinations with total bits
        max_possible = 2 ** total_bits
        actual_combinations = self.total_parameter_combinations
        
        return actual_combinations / max_possible
    
    def encode_individual_parameters(self, param_dict: Dict[str, Any]) -> str:
        """
        Encode a parameter dictionary to binary string.
        
        Args:
            param_dict: Dictionary of parameter values
            
        Returns:
            Binary string representation
        """
        binary_string = ""
        
        for param_name, encodings in self.param_binary_encodings.items():
            if param_name not in param_dict:
                raise ValueError(f"Missing parameter: {param_name}")
            
            param_value = param_dict[param_name]
            
            # Find index of this value
            try:
                value_index = encodings['values'].index(param_value)
                binary_string += encodings['binary_map'][value_index]
            except ValueError:
                raise ValueError(f"Invalid value '{param_value}' for parameter '{param_name}'")
        
        return binary_string
    
    def decode_binary_string(self, binary_string: str) -> Dict[str, Any]:
        """
        Decode binary string to parameter dictionary.
        
        Args:
            binary_string: Binary string to decode
            
        Returns:
            Dictionary of parameter values
        """
        if len(binary_string) != self._calculate_total_bits():
            raise ValueError(f"Binary string length {len(binary_string)} doesn't match expected {self._calculate_total_bits()}")
        
        decoded_params = {}
        current_pos = 0
        
        for param_name, encodings in self.param_binary_encodings.items():
            bit_length = encodings['bit_length']
            binary_value = binary_string[current_pos:current_pos + bit_length]
            
            # Convert binary to value index
            if not binary_value:
                raise ValueError(f"Empty binary value for parameter '{param_name}' at position {current_pos}")
            
            value_index = int(binary_value, 2)
            
            # Validate and clamp index
            if value_index >= len(encodings['values']):
                value_index = len(encodings['values']) - 1
            
            decoded_params[param_name] = encodings['values'][value_index]
            current_pos += bit_length
        
        return decoded_params
    
    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Parameter information dictionary
        """
        if param_name not in self.param_binary_encodings:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        encoding = self.param_binary_encodings[param_name]
        
        return {
            'name': param_name,
            'type': encoding['param_type'],
            'value_count': encoding['value_count'],
            'bit_length': encoding['bit_length'],
            'values': encoding['values'].copy(),
            'efficiency': encoding['value_count'] / (2 ** encoding['bit_length'])
        }
    
    def optimize_encoding(self) -> 'ParameterEncoder':
        """
        Create an optimized version of the encoding.
        
        Returns:
            New ParameterEncoder with optimized encoding
        """
        # For now, return self (could implement Huffman coding or other optimizations)
        return self
    
    def validate_parameter_values(self, param_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameter values against encoding.
        
        Args:
            param_dict: Parameter dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        is_valid = True
        errors = []
        
        # Check for missing parameters
        for param_name in self.param_binary_encodings.keys():
            if param_name not in param_dict:
                errors.append(f"Missing parameter: {param_name}")
                is_valid = False
        
        # Check for invalid values
        for param_name, param_value in param_dict.items():
            if param_name not in self.param_binary_encodings:
                errors.append(f"Unknown parameter: {param_name}")
                is_valid = False
            else:
                valid_values = self.param_binary_encodings[param_name]['values']
                if param_value not in valid_values:
                    errors.append(f"Invalid value '{param_value}' for parameter '{param_name}'. Valid values: {valid_values}")
                    is_valid = False
        
        return is_valid, errors
    
    def generate_random_parameters(self) -> Dict[str, Any]:
        """
        Generate a random valid parameter combination.
        
        Returns:
            Dictionary of random parameter values
        """
        import random
        
        random_params = {}
        
        for param_name, encodings in self.param_binary_encodings.items():
            random_value = random.choice(encodings['values'])
            random_params[param_name] = random_value
        
        return random_params
    
    def get_parameter_neighbors(self, param_dict: Dict[str, Any], param_name: str) -> List[Dict[str, Any]]:
        """
        Get parameter combinations that are neighbors (differ by one parameter).
        
        Args:
            param_dict: Base parameter combination
            param_name: Parameter to vary
            
        Returns:
            List of neighbor parameter combinations
        """
        if param_name not in self.param_binary_encodings:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        neighbors = []
        base_value = param_dict[param_name]
        
        for value in self.param_binary_encodings[param_name]['values']:
            if value != base_value:
                neighbor = param_dict.copy()
                neighbor[param_name] = value
                neighbors.append(neighbor)
        
        return neighbors
    
    def get_encoding_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the encoding."""
        summary = {
            'total_parameters': len(self.param_values),
            'total_bits': self._calculate_total_bits(),
            'total_combinations': self.total_parameter_combinations,
            'encoding_efficiency': self.stats['encoding_efficiency'],
            'parameters': {}
        }
        
        for param_name in self.param_binary_encodings.keys():
            summary['parameters'][param_name] = self.get_parameter_info(param_name)
        
        return summary
    
    def export_encoding(self) -> Dict[str, Any]:
        """Export encoding configuration for serialization."""
        return {
            'param_values': self.param_values,
            'param_binary_encodings': self.param_binary_encodings,
            'total_combinations': self.total_parameter_combinations,
            'statistics': self.stats
        }
    
    @classmethod
    def from_export(cls, export_data: Dict[str, Any]) -> 'ParameterEncoder':
        """Create ParameterEncoder from exported data."""
        return cls(export_data['param_values'])