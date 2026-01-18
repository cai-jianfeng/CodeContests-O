"""
Predefined configuration templates
"""

from typing import Dict, Any

# Development environment configuration
DEVELOPMENT_CONFIG: Dict[str, Any] = {
    "sandbox": {
        "hosts": ["localhost"],
        "port_range": 2,
        "max_workers_per_api": 1,
    },
    "processing": {
        "max_iterations": 2,
        "max_sample_solutions": 3,
        "sample_level_workers": 2,
        "output_generation_workers": 4,
        "solution_validation_workers": 4,
        "debug": True,
    },
    "openai": {
        "model": "gpt-4o",
        "max_tokens": 4000,
        "no_reasoning": True,
    }
}

# Production environment configuration
PRODUCTION_CONFIG: Dict[str, Any] = {
    "sandbox": {
        "hosts": ["localhost"],
        "port_range": 1,
        "max_workers_per_api": 192,
    },
    "processing": {
        "start": 0,
        "end": -1,
        "max_iterations": 4,
        "max_sample_solutions": 10,
        "use_all_solutions": True,
        "sample_level_workers": 8,
        "output_generation_workers": 192,
        "solution_validation_workers": 192,
        "debug": False,
        "only_generate": False,
        "compress_messages": True,
    },
    "openai": {
        "model": "gpt-5",
        "max_tokens": 8000,
        "no_reasoning": False,
    }
}

# Test configuration (for unit testing)
TEST_CONFIG: Dict[str, Any] = {
    "sandbox": {
        "hosts": ["localhost"],
        "port_range": 1,
        "max_workers_per_api": 1,
    },
    "processing": {
        "max_iterations": 1,
        "max_sample_solutions": 1,
        "sample_level_workers": 1,
        "output_generation_workers": 1,
        "solution_validation_workers": 1,
        "debug": True,
    },
    "openai": {
        "model": "gpt-4o",
        "max_tokens": 1000,
        "no_reasoning": True,
    }
}

# Quick validation configuration (for quick testing workflow)
QUICK_CONFIG: Dict[str, Any] = {
    "sandbox": {
        "hosts": ["localhost"],
        "port_range": 4,
        "max_workers_per_api": 4,
    },
    "processing": {
        "max_iterations": 1,
        "max_sample_solutions": 3,
        "sample_level_workers": 4,
        "output_generation_workers": 16,
        "solution_validation_workers": 16,
        "debug": False,
        "only_generate": True,
    },
    "openai": {
        "model": "gpt-4o",
        "max_tokens": 4000,
        "no_reasoning": True,
    }
}


def get_preset_config(name: str) -> Dict[str, Any]:
    """
    Get preset configuration
    
    Args:
        name: Configuration name ("development", "production", "test", "quick")
        
    Returns:
        Dict: Configuration dictionary
    """
    presets = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "test": TEST_CONFIG,
        "quick": QUICK_CONFIG,
    }
    
    if name not in presets:
        raise ValueError(f"Unknown preset config: {name}. Available: {list(presets.keys())}")
    
    return presets[name]
