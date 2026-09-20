import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)

def _get_cache_path() -> Path:
    """Return path to disk cache file."""
    from kylin_memory._vendor.kylin_agent_runtime_constants import get_hermes_home
    return get_hermes_home() / 'models_dev_cache.json'

def _load_disk_cache() -> Dict[str, Any]:
    """Load models.dev data from disk cache."""
    try:
        cache_path = _get_cache_path()
        if cache_path.exists():
            with open(cache_path, encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.debug('Failed to load models.dev disk cache: %s', e)
    return {}

