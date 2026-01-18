"""
Cloud API inference client.
Supports OpenAI, Anthropic, Google Gemini, DeepSeek, and DeepInfra.
"""
import os
import time
import json
import requests
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for a cloud API provider."""
    url: str
    key_env: str
    default_model: str
    default_workers: int = 5


# Provider configurations
PROVIDERS: Dict[str, APIConfig] = {
    # "openai": APIConfig(
    #     url="https://api.openai.com/v1/chat/completions",
    #     key_env="OPENAI_API_KEY",
    #     default_model="gpt-4o-mini",
    #     default_workers=3,
    # ),
    # "openai": APIConfig(
    #     url="https://aiberm.com/v1/chat/completions",
    #     key_env="GEMINI_3rd_PARTY_API_KEY",
    #     default_model="gemini-2.5-flash",
    #     default_workers=3,
    # ),
    # "openai": APIConfig(
    #     url="https://aiberm.com/v1/chat/completions",
    #     key_env="ANTHROPIC_3rd_PARTY_API_KEY",
    #     default_model="anthropic/claude-sonnet-4.5",
    #     default_workers=3,
    # ),
        "openai": APIConfig(
        url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        key_env="QWEN_API_KEY",
        default_model="qwen2.5-14b-instruct",
        default_workers=3,
    ),
    "anthropic": APIConfig(
        url="https://api.anthropic.com/v1/messages",
        key_env="ANTHROPIC_API_KEY",
        default_model="claude-3-5-sonnet-20241022",
        default_workers=5,
    ),
    "deepseek": APIConfig(
        url="https://api.deepseek.com/v1/chat/completions",
        key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        default_workers=3,  # Reduced for stability, especially for reasoning models
    ),
    "deepinfra": APIConfig(
        url="https://api.deepinfra.com/v1/openai/chat/completions",
        key_env="DEEPINFRA_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        default_workers=10,
    ),
    "gemini": APIConfig(
        url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        key_env="GEMINI_API_KEY",
        default_model="gemini-1.5-flash-latest",
        default_workers=5,
    ),
}


class CloudAPIClient:
    """Cloud inference API client supporting multiple providers."""
    
    def __init__(
        self,
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 5,  # Increased retries for unstable connections
        timeout: int = 180,    # Increased timeout for reasoning models (3 min)
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(PROVIDERS.keys())}")
        
        self.provider = provider
        self.config = PROVIDERS[provider]
        self.model_name = model_name or self.config.default_model
        self.api_key = api_key or os.getenv(self.config.key_env)
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._lock = threading.Lock()
        self._last_request_time: Dict[int, float] = {}
        
        if not self.api_key:
            raise ValueError(f"API key not found. Set {self.config.key_env} environment variable")
        
        logger.info(f"Initialized {provider} client with model: {self.model_name}")
    
    def _rate_limit(self, delay: float = 0.5):
        """Apply per-thread rate limiting."""
        thread_id = threading.current_thread().ident
        with self._lock:
            now = time.time()
            if thread_id in self._last_request_time:
                elapsed = now - self._last_request_time[thread_id]
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            self._last_request_time[thread_id] = time.time()
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        if self.provider == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        elif self.provider == "gemini":
            return {"Content-Type": "application/json"}
        else:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
    
    def _build_payload(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Build request payload."""
        if self.provider == "anthropic":
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        elif self.provider == "gemini":
            return {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }
        else:  # OpenAI-compatible (openai, deepseek, deepinfra)
            # GPT-5 and o-series models use max_completion_tokens instead of max_tokens
            uses_completion_tokens = self.provider == "openai" and (
                self.model_name.startswith("gpt-5") or 
                self.model_name.startswith("gpt-4.1") or
                self.model_name.startswith("gpt-4o") or
                self.model_name.startswith("o1") or 
                self.model_name.startswith("o3")
            )
            # GPT-5 doesn't support custom temperature (enforces default 1.0)
            requires_default_temp = self.provider == "openai" and (
                self.model_name.startswith("gpt-5") or
                self.model_name.startswith("o1") or
                self.model_name.startswith("o3")
            )
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            # Only add temperature if model supports it
            if not requires_default_temp:
                payload["temperature"] = temperature
            
            if uses_completion_tokens:
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
            return payload
    
    def _get_url(self) -> str:
        """Get API URL."""
        if self.provider == "gemini":
            url = self.config.url.format(model=self.model_name)
            return f"{url}?key={self.api_key}"
        return self.config.url
    
    def _extract_response(self, data: Dict[str, Any]) -> str:
        """Extract text from API response."""
        try:
            if self.provider == "anthropic":
                return data["content"][0]["text"].strip()
            elif self.provider == "gemini":
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract response: {e}")
            return ""
    
    def _is_reasoning_model(self) -> bool:
        """Check if current model is a reasoning model that needs streaming."""
        reasoning_models = ["deepseek-reasoner", "o1", "o3"]
        return any(m in self.model_name.lower() for m in reasoning_models)
    
    def _call_streaming(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        """Make streaming API call for reasoning models."""
        headers = self._build_headers()
        payload = self._build_payload(prompt, max_tokens, temperature)
        payload["stream"] = True
        url = self._get_url()
        
        try:
            response = requests.post(
                url, json=payload, headers=headers, 
                timeout=self.timeout, stream=True
            )
            
            if response.status_code != 200:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return ""
            
            # Collect streaming response
            full_content = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                        except json.JSONDecodeError:
                            continue
            
            return full_content.strip()
            
        except Exception as e:
            logger.warning(f"Streaming error: {e}")
            return ""
    
    def call(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        rate_limit_delay: float = 0.5,
    ) -> str:
        """Make API call with retries."""
        self._rate_limit(rate_limit_delay)
        
        headers = self._build_headers()
        payload = self._build_payload(prompt, max_tokens, temperature)
        url = self._get_url()
        
        # Use streaming for reasoning models to avoid connection timeouts
        use_streaming = self._is_reasoning_model() and self.provider in ["deepseek", "openai"]
        
        for attempt in range(self.max_retries):
            try:
                if use_streaming:
                    result = self._call_streaming(prompt, max_tokens, temperature)
                    if result:
                        return result
                    # Fall through to retry
                    wait = 3 ** attempt + 3
                    logger.warning(f"Streaming failed, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                
                if response.status_code == 200:
                    return self._extract_response(response.json())
                elif response.status_code == 429:  # Rate limit
                    wait = 2 ** attempt + 5
                    logger.warning(f"Rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                elif response.status_code >= 500:  # Server error
                    wait = 2 ** attempt
                    logger.warning(f"Server error {response.status_code}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"API error {response.status_code}: {response.text[:200]}")
                    return ""
                    
            except requests.exceptions.Timeout:
                wait = 2 ** attempt + 2
                logger.warning(f"Timeout, retrying in {wait}s...")
                time.sleep(wait)
            except requests.exceptions.ChunkedEncodingError as e:
                # "Response ended prematurely" - server closed connection
                wait = 3 ** attempt + 3  # Longer wait for connection issues
                logger.warning(f"Connection interrupted (ChunkedEncodingError), retrying in {wait}s...")
                time.sleep(wait)
            except requests.exceptions.RequestException as e:
                error_str = str(e)
                if "prematurely" in error_str.lower() or "incomplete" in error_str.lower():
                    # Response ended prematurely - likely server overload
                    wait = 3 ** attempt + 3
                    logger.warning(f"Response ended prematurely, retrying in {wait}s...")
                else:
                    wait = 2 ** attempt
                    logger.warning(f"Request error: {e}, retrying in {wait}s...")
                time.sleep(wait)
        
        return ""
    
    def call_with_index(
        self,
        index: int,
        prompt: str,
        **kwargs,
    ) -> Tuple[int, str]:
        """Call API and return result with index (for concurrent processing)."""
        result = self.call(prompt, **kwargs)
        return index, result


def batch_inference(
    client: CloudAPIClient,
    prompts: List[str],
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    **kwargs,
) -> List[str]:
    """
    Run batch inference with concurrent processing.
    
    Args:
        client: CloudAPIClient instance
        prompts: List of prompts to process
        max_workers: Maximum concurrent workers
        show_progress: Show progress bar
        **kwargs: Additional arguments for API call
        
    Returns:
        List of responses (in same order as prompts)
    """
    max_workers = max_workers or client.config.default_workers
    results: List[Optional[str]] = [None] * len(prompts)
    
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(prompts), desc="Processing", disable=not show_progress)
    except ImportError:
        pbar = None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(client.call_with_index, i, prompt, **kwargs): i
            for i, prompt in enumerate(prompts)
        }
        
        for future in as_completed(futures):
            try:
                idx, response = future.result()
                results[idx] = response
            except Exception as e:
                idx = futures[future]
                logger.error(f"Error processing item {idx}: {e}")
                results[idx] = ""
            
            if pbar:
                pbar.update(1)
    
    if pbar:
        pbar.close()
    
    return [r if r is not None else "" for r in results]
