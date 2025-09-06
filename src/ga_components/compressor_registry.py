"""
Compressor Registry and Factory Pattern

Eliminates code duplication in compressor configuration and creation.
Provides a clean, extensible way to register and create compressor instances.
"""

from typing import Dict, Any, Type, Callable, Optional
from abc import ABC, abstractmethod
import inspect
from ga_logging import get_logger


class CompressorFactory:
    """
    Registry-based factory for creating compressor instances and configurations.
    
    Eliminates repetitive if/elif blocks by using a registration pattern.
    """
    
    def __init__(self):
        self.logger = get_logger("CompressorFactory")
        self._compressor_registry: Dict[str, Dict[str, Any]] = {}
        self._registered_classes: Dict[str, Type] = {}
        
        # Auto-register known compressors
        self._auto_register_compressors()
    
    def _auto_register_compressors(self):
        """Automatically register known compressor classes."""
        try:
            # Import and register ZSTD compressor
            from Compressors.zstd_compressor import ZstdCompressor
            self.register_compressor(
                'ZstdCompressor',
                ZstdCompressor,
                required_args=['input_file_path', 'temp'],
                optional_args={}
            )
        except ImportError:
            self.logger.warning("ZstdCompressor not available")
        
        try:
            # Import and register LZMA compressor
            from Compressors.lzma_compressor import LzmaCompressor
            self.register_compressor(
                'LzmaCompressor',
                LzmaCompressor,
                required_args=['input_file_path', 'temp'],
                optional_args={'reference_file_path': None}
            )
        except ImportError:
            self.logger.warning("LzmaCompressor not available")
        
        try:
            # Import and register Brotli compressor
            from Compressors.brotli_compressor import BrotliCompressor
            self.register_compressor(
                'BrotliCompressor',
                BrotliCompressor,
                required_args=['input_file_path', 'temp'],
                optional_args={'reference_file_path': None}
            )
        except ImportError:
            self.logger.warning("BrotliCompressor not available")
        
        try:
            # Import and register PAQ8 compressor
            from Compressors.paq8_compressor import Paq8Compressor
            self.register_compressor(
                'Paq8Compressor',
                Paq8Compressor,
                required_args=['input_file_path', 'temp'],
                optional_args={}
            )
        except ImportError:
            self.logger.warning("Paq8Compressor not available")
        
        try:
            # Import and register AC2 compressor  
            from Compressors.ac2_compressor import Ac2Compressor
            self.register_compressor(
                'AC2Compressor',
                Ac2Compressor,
                required_args=['input_file_path', 'temp'],
                optional_args={'reference_file_path': None, 'nr_models': None}
            )
        except ImportError:
            self.logger.warning("AC2Compressor not available")
    
    def register_compressor(self, name: str, compressor_class: Type, 
                          required_args: list, optional_args: Dict[str, Any] = None):
        """
        Register a compressor class with its configuration.
        
        Args:
            name: Compressor name (e.g., 'ZstdCompressor')
            compressor_class: The compressor class
            required_args: List of required constructor arguments
            optional_args: Dict of optional arguments with default values
        """
        optional_args = optional_args or {}
        
        # Validate that the class constructor supports the specified arguments
        try:
            sig = inspect.signature(compressor_class.__init__)
            available_params = set(sig.parameters.keys()) - {'self'}
            required_set = set(required_args)
            optional_set = set(optional_args.keys())
            
            missing_required = required_set - available_params
            if missing_required:
                raise ValueError(f"Required args {missing_required} not in {name} constructor")
                
            missing_optional = optional_set - available_params  
            if missing_optional:
                self.logger.warning(f"Optional args {missing_optional} not in {name} constructor")
        
        except Exception as e:
            self.logger.error(f"Failed to validate {name} constructor", exception=e)
            return
        
        self._compressor_registry[name] = {
            'class': compressor_class,
            'required_args': required_args,
            'optional_args': optional_args
        }
        self._registered_classes[name] = compressor_class
        
        self.logger.debug(f"Registered compressor: {name}",
                         required_args=required_args,
                         optional_args=list(optional_args.keys()))
    
    def create_compressor_config(self, compressor_instance) -> Dict[str, Any]:
        """
        Create serializable configuration from a compressor instance.
        
        Args:
            compressor_instance: Instance of a registered compressor class
            
        Returns:
            Serializable configuration dictionary
        """
        compressor_type = type(compressor_instance).__name__
        
        if compressor_type not in self._compressor_registry:
            # Fallback for unregistered compressors
            return self._create_fallback_config(compressor_instance, compressor_type)
        
        registry_entry = self._compressor_registry[compressor_type]
        required_args = registry_entry['required_args']
        optional_args = registry_entry['optional_args']
        
        # Extract arguments from the instance
        config_args = {}
        
        # Extract required arguments
        for arg_name in required_args:
            if hasattr(compressor_instance, arg_name):
                config_args[arg_name] = getattr(compressor_instance, arg_name)
            else:
                self.logger.warning(f"Missing required argument: {arg_name}",
                                   compressor_type=compressor_type)
        
        # Extract optional arguments
        for arg_name, default_value in optional_args.items():
            if hasattr(compressor_instance, arg_name):
                value = getattr(compressor_instance, arg_name)
                if value is not None:  # Only include non-None optional args
                    config_args[arg_name] = value
        
        config = {
            'type': compressor_type,
            'args': config_args
        }
        
        self.logger.debug(f"Created config for {compressor_type}",
                         args=list(config_args.keys()))
        
        return config
    
    def create_compressor_from_config(self, config: Dict[str, Any]):
        """
        Create a compressor instance from a configuration.
        
        Args:
            config: Configuration dictionary with 'type' and 'args'
            
        Returns:
            Compressor instance
        """
        compressor_type = config['type']
        args = config.get('args', {})
        
        if compressor_type not in self._registered_classes:
            raise ValueError(f"Unknown compressor type: {compressor_type}")
        
        compressor_class = self._registered_classes[compressor_type]
        
        try:
            return compressor_class(**args)
        except Exception as e:
            self.logger.error(f"Failed to create {compressor_type}",
                            exception=e, args=args)
            raise
    
    def _create_fallback_config(self, compressor_instance, compressor_type: str) -> Dict[str, Any]:
        """
        Create configuration for unregistered compressor types.
        
        Uses introspection to extract common attributes.
        """
        self.logger.warning(f"Using fallback config for unregistered type: {compressor_type}")
        
        # Common attributes found in most compressors
        common_attrs = ['input_file_path', 'temp', 'reference_file_path', 'nr_models']
        
        config_args = {}
        for attr in common_attrs:
            if hasattr(compressor_instance, attr):
                value = getattr(compressor_instance, attr)
                if value is not None:
                    config_args[attr] = value
        
        return {
            'type': compressor_type,
            'args': config_args
        }
    
    def get_registered_compressors(self) -> list:
        """Get list of registered compressor names."""
        return list(self._compressor_registry.keys())
    
    def is_registered(self, compressor_name: str) -> bool:
        """Check if a compressor is registered."""
        return compressor_name in self._compressor_registry
    
    def get_compressor_info(self, compressor_name: str) -> Dict[str, Any]:
        """Get registration information for a compressor."""
        if compressor_name not in self._compressor_registry:
            return {}
        
        entry = self._compressor_registry[compressor_name]
        return {
            'class_name': entry['class'].__name__,
            'module': entry['class'].__module__,
            'required_args': entry['required_args'],
            'optional_args': list(entry['optional_args'].keys())
        }


# Global factory instance
_compressor_factory: Optional[CompressorFactory] = None


def get_compressor_factory() -> CompressorFactory:
    """Get or create the global compressor factory instance."""
    global _compressor_factory
    if _compressor_factory is None:
        _compressor_factory = CompressorFactory()
    return _compressor_factory


def create_compressor_config(compressor_instance) -> Dict[str, Any]:
    """
    Convenience function to create compressor configuration.
    
    Args:
        compressor_instance: Compressor instance
        
    Returns:
        Serializable configuration dictionary
    """
    factory = get_compressor_factory()
    return factory.create_compressor_config(compressor_instance)


def create_compressor_from_config(config: Dict[str, Any]):
    """
    Convenience function to create compressor from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compressor instance
    """
    factory = get_compressor_factory()
    return factory.create_compressor_from_config(config)