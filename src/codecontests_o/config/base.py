"""
Configuration Management Module - Define all configuration data classes
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class OpenAIConfig:
    """OpenAI API Configuration"""
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    max_tokens: int = 8000
    no_reasoning: bool = True
    max_attempts: int = 3
    timeout: int = 400
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        if not self.api_key:
            errors.append("OpenAI API key is not set")
        if not self.api_base:
            errors.append("OpenAI API base URL is not set")
        return errors


@dataclass
class SandboxConfig:
    """Sandbox API Configuration"""
    hosts: List[str] = field(default_factory=lambda: ["localhost"])
    base_port: int = 8080
    port_range: int = 4
    max_workers_per_api: int = 1
    compile_timeout: int = 20
    run_timeout: int = 20
    
    def get_api_paths(self) -> List[str]:
        """Generate all API paths"""
        api_paths = []
        
        for host in self.hosts:
            for i in range(self.port_range):
                port = self.base_port + i
                for _ in range(self.max_workers_per_api):
                    api_paths.append(f"http://{host}:{port}/")
        
        return api_paths
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        if not self.hosts:
            errors.append("No sandbox hosts configured")
        return errors


@dataclass
class ProcessingConfig:
    """Processing Configuration"""
    start: int = 0
    end: int = -1  # -1 means process until the end
    max_iterations: int = 3
    max_sample_solutions: int = 10
    use_all_solutions: bool = False
    debug: bool = False
    save_intermediate_results: bool = True
    
    # Parallel configuration
    sample_level_workers: int = 4
    output_generation_workers: int = 4
    solution_validation_workers: int = 4
    
    # Feature flags
    only_generate: bool = False  # Only generate, do not validate
    compress_messages: bool = True  # Compress messages to save tokens
    
    # Truncated length
    truncated_length: int = 1000


@dataclass
class DatasetConfig:
    """Dataset Configuration"""
    data_path: str = ""
    split: str = "test"
    dataset_type: str = "code_contests"
    results_dir: str = "./results"
    solution_results_dir: str = ""
    re_tle_file: str = ""
    
    def __post_init__(self):
        if not self.solution_results_dir:
            self.solution_results_dir = os.path.join(self.results_dir, "solution_results")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.solution_results_dir, exist_ok=True)
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        if not self.data_path:
            errors.append("Dataset data_path is not set")
        elif not os.path.exists(self.data_path):
            errors.append(f"Dataset path does not exist: {self.data_path}")
        return errors


@dataclass
class ResourceConfig:
    """Resource File Configuration"""
    testlib_path: str = "testlib.h"
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        if not os.path.exists(self.testlib_path):
            errors.append(f"testlib.h not found at: {self.testlib_path}")
        return errors


@dataclass
class Config:
    """
    Full Configuration
    
    Aggregates all sub-configurations, provides unified configuration access interface
    """
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        errors.extend(self.openai.validate())
        errors.extend(self.sandbox.validate())
        errors.extend(self.dataset.validate())
        errors.extend(self.resource.validate())
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information"""
        api_paths = self.sandbox.get_api_paths()
        
        return {
            "api_endpoints": len(api_paths),
            "total_workers": len(api_paths),
            "dataset_path": self.dataset.data_path,
            "results_dir": self.dataset.results_dir,
            "max_iterations": self.processing.max_iterations,
            "model": self.openai.model,
            "sample_workers": self.processing.sample_level_workers,
            "output_workers": self.processing.output_generation_workers,
            "validation_workers": self.processing.solution_validation_workers,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        config = cls()
        
        if "openai_config" in config_dict or "openai" in config_dict:
            openai_dict = config_dict.get("openai_config", config_dict.get("openai", {}))
            for key, value in openai_dict.items():
                if hasattr(config.openai, key):
                    setattr(config.openai, key, value)
        
        if "sandbox_config" in config_dict or "sandbox" in config_dict:
            sandbox_dict = config_dict.get("sandbox_config", config_dict.get("sandbox", {}))
            for key, value in sandbox_dict.items():
                if hasattr(config.sandbox, key):
                    setattr(config.sandbox, key, value)
        
        if "processing_config" in config_dict or "processing" in config_dict:
            processing_dict = config_dict.get("processing_config", config_dict.get("processing", {}))
            for key, value in processing_dict.items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
        
        if "dataset_config" in config_dict or "dataset" in config_dict:
            dataset_dict = config_dict.get("dataset_config", config_dict.get("dataset", {}))
            for key, value in dataset_dict.items():
                if hasattr(config.dataset, key):
                    setattr(config.dataset, key, value)
        
        if "resource_config" in config_dict or "resource" in config_dict:
            resource_dict = config_dict.get("resource_config", config_dict.get("resource", {}))
            for key, value in resource_dict.items():
                if hasattr(config.resource, key):
                    setattr(config.resource, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "openai": {
                "api_base": self.openai.api_base,
                "model": self.openai.model,
                "max_tokens": self.openai.max_tokens,
                "no_reasoning": self.openai.no_reasoning,
                "max_attempts": self.openai.max_attempts,
            },
            "sandbox": {
                "hosts": self.sandbox.hosts,
                "base_port": self.sandbox.base_port,
                "port_range": self.sandbox.port_range,
                "max_workers_per_api": self.sandbox.max_workers_per_api,
            },
            "processing": {
                "start": self.processing.start,
                "end": self.processing.end,
                "max_iterations": self.processing.max_iterations,
                "max_sample_solutions": self.processing.max_sample_solutions,
                "sample_level_workers": self.processing.sample_level_workers,
                "output_generation_workers": self.processing.output_generation_workers,
                "solution_validation_workers": self.processing.solution_validation_workers,
            },
            "dataset": {
                "data_path": self.dataset.data_path,
                "split": self.dataset.split,
                "results_dir": self.dataset.results_dir,
            }
        }
