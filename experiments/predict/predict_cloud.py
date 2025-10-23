#!/usr/bin/env python3
"""
Cloud inference API for baseline prediction using various providers
Supports OpenAI, Anthropic Claude, Google Gemini, DeepSeek, and DeepInfra
Features concurrent processing for improved performance
"""
import json
import os
import sys
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    tqdm = None

# Load .env file
def load_env():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), "../../../../.env")
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from experiments.data_process.data_utils import extract_sql_prompt_dataset, extract_sql_role_prompt_dataset
from experiments.predict.response_cleaner import clean_model_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudAPIConfig:
    """Configuration for different cloud API providers"""
    
    PROVIDERS = {
        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "key_env": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini",
            "models": [
                "gpt-5",
                "gpt-5-mini",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4o-realtime-preview",  # audio/stream capable models still usable via chat API
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
            ],
            "default_workers": 3  # Conservative default, users can override
        },
        "anthropic": {
            "url": "https://api.anthropic.com/v1/messages",
            "key_env": "ANTHROPIC_API_KEY",
            "default_model": "claude-sonnet-4-5",
            "models": [
                "claude-sonnet-4-5",
                "claude-sonnet-4",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            "default_workers": 5  # Conservative default for Anthropic
        },
        "deepseek": {
            "url": "https://api.deepseek.com/v1/chat/completions",
            "key_env": "DEEPSEEK_API_KEY", 
            "default_model": "deepseek-chat",
            "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
            "default_workers": 5  # Conservative default, users can override
        },
        "deepinfra": {
            "url": "https://api.deepinfra.com/v1/openai/chat/completions",
            "key_env": "DEEPINFRA_API_KEY",
            "default_model": "google/gemma-3-4b-it",
            "models": [
                "google/gemma-3-4b-it",
                "google/gemma-3-27b-it",
                "meta-llama/Meta-Llama-3.1-405B-Instruct",
                "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "microsoft/WizardLM-2-8x22B",
                "mistralai/Mixtral-8x22B-Instruct-v0.1",
            ],
            "default_workers": 10  # DeepInfra allows higher concurrency
        },
        "gemini": {
            "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "key_env": "GEMINI_API_KEY",
            "default_model": "gemini-2.0-flash-exp",
            "models": [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash-002",
                "gemini-1.5-pro-002",
            ],
            "default_workers": 5  # Gemini allows moderate concurrency
        }
    }
    
    @classmethod
    def get_provider_config(cls, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(cls.PROVIDERS.keys())}")
        return cls.PROVIDERS[provider]
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers"""
        return list(cls.PROVIDERS.keys())
    
    @classmethod
    def get_default_workers(cls, provider: str) -> int:
        """Get conservative default worker count for provider"""
        config = cls.get_provider_config(provider)
        return config["default_workers"]

    OPENAI_COMPLETION_TOKEN_PREFIXES = ("gpt-4.1", "gpt-4o", "gpt-5")
    OPENAI_FIXED_TEMPERATURE_PREFIXES = ("gpt-5", "gpt-5-mini") # "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", 

    @classmethod
    def uses_completion_tokens(cls, provider: str, model_name: Optional[str]) -> bool:
        if provider != "openai" or not model_name:
            return False
        return any(model_name.startswith(prefix) for prefix in cls.OPENAI_COMPLETION_TOKEN_PREFIXES)

    @classmethod
    def requires_default_temperature(cls, provider: str, model_name: Optional[str]) -> bool:
        if provider != "openai" or not model_name:
            return False
        return any(model_name.startswith(prefix) for prefix in cls.OPENAI_FIXED_TEMPERATURE_PREFIXES)

class CloudInferenceClient:
    """Cloud inference API client supporting multiple providers with concurrent processing"""
    
    def __init__(self, provider: str, model_name: Optional[str] = None, api_key: Optional[str] = None, 
                 max_retries: int = 3, request_timeout: int = 60):
        self.provider = provider
        self.config = CloudAPIConfig.get_provider_config(provider)
        self.model_name = model_name or self.config["default_model"]
        self.api_key = api_key or os.getenv(self.config["key_env"])
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._lock = threading.Lock()
        self._last_request_time = {}
        
        if not self.api_key:
            raise ValueError(f"API key not found. Set {self.config['key_env']} environment variable")
        
        logger.info(f"Initialized {provider} client with model: {self.model_name}")
    
    def _apply_rate_limit(self, rate_limit_delay: float = 1.0):
        """Apply rate limiting per thread"""
        thread_id = threading.current_thread().ident
        
        with self._lock:
            current_time = time.time()
            if thread_id in self._last_request_time:
                time_since_last = current_time - self._last_request_time[thread_id]
                if time_since_last < rate_limit_delay:
                    sleep_time = rate_limit_delay - time_since_last
                    time.sleep(sleep_time)
            self._last_request_time[thread_id] = time.time()
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers for the API provider"""
        if self.provider == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
        elif self.provider == "gemini":
            # Gemini uses API key as query parameter, not in headers
            return {
                "Content-Type": "application/json"
            }
        else:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build request payload for the API provider"""
        # Anthropic uses a different message format
        if self.provider == "anthropic":
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            # Note: Anthropic doesn't allow both temperature and top_p
            # We use only temperature for sampling control (set later in this function)
        elif self.provider == "gemini":
            # Gemini uses a different format with contents array
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            # Gemini-specific generation config will be added below
        else:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            # Add provider-specific parameters
            if self.provider == "openai":
                payload.update({
                    "top_p": kwargs.get("top_p", 1.0),
                    "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                    "presence_penalty": kwargs.get("presence_penalty", 0.0)
                })
            elif self.provider == "deepseek":
                payload.update({
                    "top_p": kwargs.get("top_p", 0.95),
                    "stream": False
                })
            elif self.provider == "deepinfra":
                payload.update({
                    "top_p": kwargs.get("top_p", 0.9),
                    "stream": False
                })
            
        temperature_raw = kwargs.get("temperature")
        temperature_value: Optional[float] = None
        if temperature_raw is not None and str(temperature_raw).strip() != "":
            try:
                temperature_value = float(temperature_raw)
            except (TypeError, ValueError):
                logger.warning(f"Invalid temperature value provided: {temperature_raw}, ignoring")

        if CloudAPIConfig.requires_default_temperature(self.provider, self.model_name):
            if temperature_value is not None and abs(temperature_value - 1.0) > 1e-6:
                logger.warning(
                    "Model %s enforces temperature=1.0; overriding requested %.3f",
                    self.model_name,
                    temperature_value,
                )
            # Do not send temperature parameter; API default (1.0) will be used.
        else:
            if temperature_value is None:
                # Backwards-compatible default for deterministic decoding on legacy models
                if self.provider == "anthropic":
                    temperature_value = 1.0  # Anthropic's default
                elif self.provider == "gemini":
                    temperature_value = 1.0  # Gemini's default
                else:
                    temperature_value = 0.0
            
            if self.provider == "gemini":
                # Gemini uses generationConfig for parameters
                if "generationConfig" not in payload:
                    payload["generationConfig"] = {}
                payload["generationConfig"]["temperature"] = temperature_value
            else:
                payload["temperature"] = temperature_value

        max_tokens_raw = kwargs.get("max_tokens")
        max_tokens_value: Optional[int] = None
        if max_tokens_raw is not None and str(max_tokens_raw).strip() != "":
            try:
                max_tokens_value = int(float(max_tokens_raw))
            except (TypeError, ValueError):
                logger.warning(f"Invalid max_tokens value provided: {max_tokens_raw}, ignoring")

        if max_tokens_value is not None:
            if CloudAPIConfig.uses_completion_tokens(self.provider, self.model_name):
                payload["max_completion_tokens"] = max_tokens_value
            elif self.provider == "gemini":
                # Gemini uses generationConfig for maxOutputTokens
                if "generationConfig" not in payload:
                    payload["generationConfig"] = {}
                payload["generationConfig"]["maxOutputTokens"] = max_tokens_value
            else:
                payload["max_tokens"] = max_tokens_value
        elif self.provider == "anthropic":
            # Anthropic requires max_tokens parameter
            payload["max_tokens"] = 4096
        elif self.provider == "gemini":
            # Gemini default max output tokens
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["maxOutputTokens"] = 8192
        elif self.provider != "openai":
            # Provide a safe default for non-OpenAI providers when not specified
            payload.setdefault("max_tokens", 1024)
        
        return payload
    
    def _extract_response(self, response_data: Dict[str, Any]) -> str:
        """Extract text response from API response"""
        try:
            if self.provider == "anthropic":
                # Anthropic response format: {"content": [{"type": "text", "text": "..."}], ...}
                return response_data["content"][0]["text"].strip()
            elif self.provider == "gemini":
                # Gemini response format: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
                return response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                # OpenAI-compatible format
                return response_data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract response: {e}")
            return "Error: Unable to extract response from API"
    
    def call_api(self, prompt: str, rate_limit_delay: float = 1.0, **kwargs) -> str:
        """Make API call with retries and robust error handling"""
        # Apply rate limiting
        self._apply_rate_limit(rate_limit_delay)
        
        headers = self._build_headers()
        payload = self._build_payload(prompt, **kwargs)
        
        # Build the API URL
        if self.provider == "gemini":
            # Gemini uses model name in URL and API key as query parameter
            api_url = self.config["url"].format(model=self.model_name)
            api_url = f"{api_url}?key={self.api_key}"
        else:
            api_url = self.config["url"]
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    api_url, 
                    json=payload, 
                    headers=headers, 
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._extract_response(result)
                elif response.status_code == 429:  # Rate limit exceeded
                    wait_time = 2 ** attempt + 5  # Longer wait for rate limits
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        return "Error: API rate limit exceeded after retries"
                elif response.status_code in [500, 502, 503, 504]:  # Server errors
                    wait_time = 2 ** attempt
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        return f"Error: Server error {response.status_code} after retries"
                else:
                    logger.warning(f"API Error {response.status_code}: {response.text[:200]}")
                    return f"Error: API error {response.status_code}"
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request timeout, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return "Error: Request timeout after retries"
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Connection error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return "Error: Connection error after retries"
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return f"Error: Unexpected error - {str(e)}"
        
        return "Error: All retry attempts failed"
    
    def call_api_with_index(self, index: int, prompt: str, rate_limit_delay: float = 1.0, **kwargs) -> Tuple[int, str]:
        """Make API call with index for concurrent processing"""
        result = self.call_api(prompt, rate_limit_delay, **kwargs)
        return (index, result)

class ProgressTracker:
    """Thread-safe progress tracking with progress bar"""
    
    def __init__(self, total: int, use_progress_bar: bool = True):
        self.total = total
        self.completed = 0
        self.errors = 0
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.use_progress_bar = use_progress_bar and tqdm is not None
        
        if self.use_progress_bar:
            try:
                self.pbar = tqdm(
                    total=total,
                    desc="Processing",
                    unit="samples",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Errors: {postfix}"
                )
                self.pbar.set_postfix({"errors": 0})
            except Exception as e:
                logger.warning(f"Could not initialize progress bar: {e}, falling back to logging")
                self.use_progress_bar = False
                self.pbar = None
        
        if not self.use_progress_bar:
            self.pbar = None
            logger.info(f"Starting processing of {total} samples...")
    
    def update(self, success: bool = True):
        """Update progress counter"""
        with self._lock:
            self.completed += 1
            if not success:
                self.errors += 1
            
            if self.use_progress_bar and self.pbar:
                self.pbar.update(1)
                self.pbar.set_postfix({"errors": self.errors})
            else:
                # Fallback to logging with more frequent updates for better feedback
                if self.completed % 5 == 0 or self.completed in [1, 2, 3] or self.completed == self.total:
                    elapsed = time.time() - self.start_time
                    rate = self.completed / elapsed if elapsed > 0 else 0
                    remaining = (self.total - self.completed) / rate if rate > 0 else 0
                    
                    progress_str = (
                        f"Progress: {self.completed}/{self.total} "
                        f"({self.completed/self.total*100:.1f}%) "
                        f"Errors: {self.errors} "
                        f"Rate: {rate:.2f}/s"
                    )
                    
                    if remaining > 0:
                        progress_str += f" ETA: {remaining:.0f}s"
                    
                    logger.info(progress_str)
    
    def close(self):
        """Close progress bar if it exists"""
        if self.pbar:
            self.pbar.close()

def cloud_inference_predict(
    input_file: str,
    output_file: str,
    provider: str = "openai",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    workers: Optional[int] = None,
    rate_limit_delay: float = 1.0,
    **api_kwargs
) -> Dict[str, Any]:
    """
    Run cloud inference prediction with concurrent processing
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output SQL file
        provider: API provider (openai, deepseek)
        model_name: Model name (optional, uses provider default)
        api_key: API key (optional, uses env variable)
        max_samples: Maximum number of samples to process (optional)
        workers: Number of concurrent workers (optional, uses provider recommended)
        rate_limit_delay: Base delay between requests in seconds
        **api_kwargs: Additional API parameters (temperature, max_tokens, etc.)
    
    Returns:
        Dictionary with prediction statistics
    """
    
    logger.info(f"Starting cloud inference with provider: {provider}")
    
    # Determine worker count
    if workers is None:
        workers = CloudAPIConfig.get_default_workers(provider)
        logger.info(f"Using default {workers} concurrent workers for {provider}")
    else:
        logger.info(f"Using {workers} concurrent workers (user specified)")
        logger.warning(f"Note: Make sure your API key supports {workers} concurrent requests to avoid rate limiting")
    
    # Load and validate data
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input file {input_file}: {e}")
        raise
    
    # Limit samples if specified
    if max_samples and max_samples < len(data):
        data = data[:max_samples]
        logger.info(f"Limited to first {max_samples} samples")
    
    # Prepare data based on format
    try:
        if data and isinstance(data[0], dict) and "role" in data[0]:
            predict_data = []
            for i, item in enumerate(data):
                try:
                    processed_item = extract_sql_role_prompt_dataset(item)
                    if isinstance(processed_item, dict) and "input" in processed_item:
                        predict_data.append(processed_item)
                    else:
                        logger.warning(f"Item {i} processed incorrectly, skipping")
                except Exception as e:
                    logger.error(f"Failed to process role-based item {i}: {e}")
                    continue
            data_type = "role-based"
        else:
            predict_data = []
            for i, item in enumerate(data):
                try:
                    processed_item = extract_sql_prompt_dataset(item)
                    if isinstance(processed_item, dict) and "input" in processed_item:
                        predict_data.append(processed_item)
                    else:
                        logger.warning(f"Item {i} processed incorrectly, skipping")
                except Exception as e:
                    logger.error(f"Failed to process standard item {i}: {e}")
                    continue
            data_type = "standard"
        
        if not predict_data:
            raise ValueError("No valid data items could be processed")
            
        logger.info(f"Loaded {len(predict_data)} {data_type} samples")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise
    
    # Initialize API client
    try:
        client = CloudInferenceClient(provider, model_name, api_key)
    except Exception as e:
        logger.error(f"Failed to initialize API client: {e}")
        raise
    
    # Initialize progress tracking
    progress = ProgressTracker(len(predict_data))
    results_dict = {}  # Use dict to store results with index as key
    start_time = time.time()
    
    # Calculate per-worker rate limit to distribute overall rate across workers
    per_worker_delay = rate_limit_delay
    
    try:
        # Run concurrent predictions
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks with indexed keys for reordering
            future_to_index = {
                executor.submit(
                    client.call_api_with_index, 
                    i, 
                    item["input"], 
                    per_worker_delay,
                    **api_kwargs
                ): i 
                for i, item in enumerate(predict_data)
            }
            
            # Collect results as they complete with timeout handling
            completed_futures = []
            for future in as_completed(future_to_index, timeout=None):
                try:
                    index, result = future.result(timeout=120)  # 2 minute timeout per task
                    results_dict[index] = result
                    
                    # Only count actual system/API errors, not model's "Sorry" responses
                    success = not result.startswith("Error:")
                    progress.update(success)
                    completed_futures.append(future)
                    
                except Exception as e:
                    index = future_to_index[future]
                    logger.error(f"Task {index} failed: {e}")
                    results_dict[index] = f"Error: Task execution failed - {str(e)}"
                    progress.update(False)
                    completed_futures.append(future)
            
    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, stopping...")
        return {"error": "Interrupted by user"}
    except Exception as e:
        logger.error(f"Concurrent processing failed: {e}")
        return {"error": str(e)}
    finally:
        progress.close()
    
    # Reorder results by index to maintain original sequence
    results = []
    for i in range(len(predict_data)):
        if i in results_dict:
            results.append(results_dict[i])
        else:
            results.append("Error: No result")
    
    # Save results
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            for idx, result in enumerate(results):
                # Clean and format output
                cleaned_result = clean_model_response(result)
                if not cleaned_result:
                    cleaned_result = "Empty response"
                results[idx] = cleaned_result
                f.write(cleaned_result + "\n")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
        raise
    
    # Calculate statistics
    total_time = time.time() - start_time
    
    # Count different types of responses
    error_count = sum(1 for result in results if result.startswith("Error:"))
    sorry_count = sum(1 for result in results if "Sorry" in result and not result.startswith("Error:"))
    success_count = len(results) - error_count - sorry_count
    
    stats = {
        "provider": provider,
        "model": client.model_name,
        "total_samples": len(predict_data),
        "successful_samples": success_count,
        "sorry_responses": sorry_count, 
        "system_errors": error_count,
        "workers": workers,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_sample": round(total_time / len(predict_data), 2),
        "throughput_samples_per_second": round(len(predict_data) / total_time, 2),
        "output_file": output_file
    }
    
    logger.info(f"Cloud inference completed successfully")
    logger.info(f"Statistics: {stats}")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Concurrent Cloud Inference for Text2SQL Baseline")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output SQL file path") 
    parser.add_argument("--provider", default="openai", choices=CloudAPIConfig.get_available_providers(),
                       help="API provider")
    parser.add_argument("--model", help="Model name (uses provider default if not specified)")
    parser.add_argument("--api_key", help="API key (uses environment variable if not specified)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--workers", type=int, help="Number of concurrent workers (uses provider default if not specified)")
    parser.add_argument("--rate_limit", type=float, default=1.0, 
                       help="Base rate limit delay between requests (seconds)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens in response")
    
    args = parser.parse_args()
    
    # Run cloud inference
    api_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens
    }
    
    try:
        stats = cloud_inference_predict(
            input_file=args.input,
            output_file=args.output,
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            max_samples=args.max_samples,
            workers=args.workers,
            rate_limit_delay=args.rate_limit,
            **api_kwargs
        )
        
        print(f"Prediction completed successfully!")
        print(f"Results saved to: {stats['output_file']}")
        print(f"Successful SQL: {stats['successful_samples']}/{stats['total_samples']} samples")
        if stats['sorry_responses'] > 0:
            print(f"Role-based denials: {stats['sorry_responses']} samples")
        if stats['system_errors'] > 0:
            print(f"System errors: {stats['system_errors']} samples")
        print(f"Used {stats['workers']} concurrent workers")
        print(f"Throughput: {stats['throughput_samples_per_second']:.2f} samples/second")
        print(f"Total time: {stats['total_time_seconds']}s")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)
