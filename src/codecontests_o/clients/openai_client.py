"""
OpenAI API Client Wrapper
"""

from typing import List, Dict, Union, Optional
from pydantic import BaseModel
from openai import OpenAI


class InitCommandModel(BaseModel):
    """Initial command generation model"""
    input_constraints_summary: str
    command_list: List[str]
    search_replace_generator_blocks: List[str] = []
    generator: Optional[str] = None


class CommandModel(BaseModel):
    """Iterative command generation model"""
    replace_command_list: List[str]
    add_command_list: List[str]
    search_replace_generator_blocks: List[str] = []


# List of models supporting reasoning mode
REASONING_MODELS = ["o3-mini", "o4-mini", "gpt-5"]
NO_REASONING_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]


class OpenAIClient:
    """
    OpenAI API Client Wrapper
    
    Provides interface for command generation, supporting reasoning and non-reasoning modes
    """
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 8000,
        no_reasoning: bool = True,
        max_attempts: int = 3,
        timeout: int = 400
    ):
        """
        Initialize OpenAI Client
        
        Args:
            api_base: API base URL
            api_key: API key
            model: Model name
            max_tokens: Max tokens
            no_reasoning: Disable reasoning mode
            max_attempts: Max retry attempts
            timeout: Request timeout
        """
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.no_reasoning = no_reasoning
        self.max_attempts = max_attempts
        self.timeout = timeout
        
        # Validate model configuration
        if no_reasoning and model in REASONING_MODELS:
            print(f"Warning: Model {model} supports reasoning but no_reasoning is True")
        elif not no_reasoning and model in NO_REASONING_MODELS:
            raise ValueError(f"Model {model} does not support reasoning mode")
    
    def generate_command(
        self,
        messages: List[Dict],
        is_first: bool,
        sample_id: str = "",
        logger=None
    ) -> Union[InitCommandModel, CommandModel, str, None]:
        """
        Generate commands using OpenAI API
        
        Args:
            messages: Conversation messages
            is_first: Is first generation (determines return model type)
            sample_id: Sample ID (for logging)
            logger: Logger
            
        Returns:
            InitCommandModel or CommandModel or "Exceeded" (context exceeded) or None (failed)
        """
        command_model = InitCommandModel if is_first else CommandModel
        
        for attempt in range(self.max_attempts):
            try:
                if self.no_reasoning:
                    response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        response_format=command_model,
                        timeout=self.timeout,
                    )
                else:
                    response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        response_format=command_model,
                        reasoning_effort="low",
                        verbosity="medium",
                        timeout=self.timeout,
                    )
                
                return response.choices[0].message.parsed
                
            except Exception as e:
                error_msg = f"Error generating command (attempt {attempt + 1}): {e}"
                if logger:
                    logger(sample_id, error_msg)
                else:
                    print(f"[{sample_id}] {error_msg}")
                
                # Check if context length exceeded
                if hasattr(e, 'status_code') and e.status_code == 413:
                    return "Exceeded"
                if hasattr(e, 'code') and e.code == 'context_length_exceeded':
                    return "Exceeded"
        
        return None
    
    def chat(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Simple chat interface
        
        Args:
            messages: Conversation messages
            max_tokens: Max tokens
            temperature: Sampling temperature
            
        Returns:
            str: Response content, None if failed
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
                timeout=self.timeout,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Chat error: {e}")
            return None
