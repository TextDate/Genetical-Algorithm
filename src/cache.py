# Cache module - Import from the correct multiprocess cache
from multiprocess_cache import get_global_cache

# Re-export for compatibility
__all__ = ['get_global_cache']