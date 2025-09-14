"""
Main Oracle class for handling LLM queries
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai
from openai import OpenAI
import google as genai
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from .config.models import (
    all_model_list,
    openai_model_list,
    google_model_list,
    azure_model_list,
    deepinfra_model_list,
    deepseek_model_list,
    DEEP_INFRA_BASE_URL,
    DEEPSEEK_BASE_URL,
    DEEP_INFRA_MAP,
)
from .utils.clients import (
    openai_chat_query,
    openai_embed_query,
    google_chat_query,
    google_embed_query,
    azure_chat_query,
    deepseek_chat_query,
)

class Oracle:
    def __init__(self, model, apikey=None, azure_end_point=''):
        """
        Initialize Oracle instance with specified model and API key
        """
        assert model in all_model_list, f'err: model named {model} is not supported'
        self.model = model
        self.apikey = apikey

        if model in openai_model_list:
            if apikey:
                openai.api_key = apikey
            self.client = OpenAI()
        elif model in deepinfra_model_list:
            self.client = OpenAI(api_key=apikey, base_url=DEEP_INFRA_BASE_URL)
        elif model in google_model_list:
            self.client = genai.Client(api_key=apikey)
        elif model in azure_model_list:
            azure_credential = AzureKeyCredential(apikey)
            self.client = ChatCompletionsClient(endpoint=azure_end_point, credential=azure_credential)
        elif model in deepseek_model_list:
            self.client = OpenAI(api_key=apikey, base_url=DEEPSEEK_BASE_URL)
    
    def query(self, prompt_sys, prompt_user, temp=1.0, top_p=0.9, query_key=None, max_completion_tokens=1000):
        """
        Execute a single chat completion query
        """
        if self.model in openai_model_list:
            return openai_chat_query(
                self.client, self.model, prompt_sys, prompt_user,
                temp, top_p, max_completion_tokens, query_key
            )
        elif self.model in deepinfra_model_list:
            return openai_chat_query(
                self.client, DEEP_INFRA_MAP[self.model], prompt_sys,
                prompt_user, temp, top_p, max_completion_tokens, query_key
            )
        elif self.model in google_model_list:
            return google_chat_query(
                self.client, self.model, prompt_sys,
                prompt_user, temp, top_p, query_key
            )
        elif self.model in azure_model_list:
            return azure_chat_query(
                self.client, self.model, prompt_sys,
                prompt_user, temp, top_p, query_key
            )
        elif self.model in deepseek_model_list:
            return deepseek_chat_query(
                self.client, self.model, prompt_sys,
                prompt_user, temp, top_p, max_completion_tokens, query_key
            )
    
    def query_all(self, prompt_sys, prompt_user_all, workers=12, temp=1.0, top_p=0.9,
                 query_key_list=[], reorder=True, max_completion_tokens=1000, **kwargs):
        """
        Execute multiple chat completion queries in parallel
        """
        if isinstance(prompt_sys, str):
            prompt_sys = [prompt_sys] * len(prompt_user_all)

        results = []
        print(f"Total queries: {len(prompt_user_all)}, start collecting...")

        model_name = self.model
        if self.model in openai_model_list:
            single_query_fn = openai_chat_query
        elif self.model in deepinfra_model_list:
            single_query_fn = openai_chat_query
            model_name = DEEP_INFRA_MAP[self.model]
        elif self.model in google_model_list:
            single_query_fn = google_chat_query
        elif self.model in azure_model_list:
            single_query_fn = azure_chat_query
        elif self.model in deepseek_model_list:
            single_query_fn = deepseek_chat_query
        
        if reorder:
            if query_key_list:
                indexed_query_key_list = [f"{i:06d}_{key}" for i, key in enumerate(query_key_list)]
            else:
                indexed_query_key_list = [f"{i:06d}" for i in range(len(prompt_user_all))]
        else:
            indexed_query_key_list = query_key_list if query_key_list else []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            if indexed_query_key_list:
                if single_query_fn is openai_chat_query:
                    future_results = [
                        executor.submit(single_query_fn, self.client, model_name,
                                     prompt_sys[i], p, temp, top_p,
                                     max_completion_tokens, indexed_query_key_list[i])
                        for i, p in enumerate(prompt_user_all)
                    ]
                else:
                    future_results = [
                        executor.submit(single_query_fn, self.client, model_name,
                                     prompt_sys[i], p, temp, top_p,
                                     indexed_query_key_list[i])
                        for i, p in enumerate(prompt_user_all)
                    ]
            else:
                if single_query_fn is openai_chat_query:
                    future_results = [
                        executor.submit(single_query_fn, self.client, model_name,
                                     prompt_sys[i], p, temp, top_p,
                                     max_completion_tokens)
                        for i, p in enumerate(prompt_user_all)
                    ]
                else:
                    future_results = [
                        executor.submit(single_query_fn, self.client, model_name,
                                     prompt_sys[i], p, temp, top_p)
                        for i, p in enumerate(prompt_user_all)
                    ]
        
            for future in tqdm(as_completed(future_results),
                             total=len(prompt_user_all),
                             desc="Processing Items"):
                result = future.result()
                results.append(result)
                
        if reorder:
            results.sort(key=lambda x: int(x["query"].split("_")[0]))
            for result in results:
                if "_" in result["query"]:
                    result["query"] = "_".join(result["query"].split("_")[1:])

        return results
    
    def encode(self, text, dim=1024):
        """
        Generate embeddings for a single text
        """
        if self.model in openai_model_list:
            return openai_embed_query(self.client, self.model, [text], dim)
        elif self.model in google_model_list:
            return google_embed_query(self.client, self.model, [text], dim)
    
    def quick_encode(self, text, dim=1024):
        """
        Generate embeddings for a single text and return directly
        """
        if self.model in openai_model_list:
            return openai_embed_query(
                self.client, self.model, [text],
                dim, original_key=True
            )[text]
        else:
            raise RuntimeError(f'unsupported error for model {self.model}')

    def encode_all(self, list_of_text, workers=12, dim=1024, chunk_size=100, original_key=False):
        """
        Generate embeddings for multiple texts in parallel
        """
        sublists = [list_of_text[i:i + chunk_size] for i in range(0, len(list_of_text), chunk_size)]
        
        results = []
        print(f"Total queries: {len(list_of_text)}, start collecting...")
        
        if self.model in openai_model_list:
            single_query_fn = openai_embed_query
        elif self.model in google_model_list:
            single_query_fn = google_embed_query
        else:
            raise RuntimeError(f'unsupported error for model {self.model}')

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_results = [
                executor.submit(single_query_fn, self.client,
                              self.model, p, dim, original_key)
                for p in sublists
            ]
        
            for future in tqdm(as_completed(future_results),
                             total=len(sublists),
                             desc="Processing Items"):
                result = future.result()
                results.append(result)
        return results

    def collect_results(self, results):
        """
        Collect results from multiple queries into a single dictionary
        """
        res_dict = {}
        for result in results:
            res_dict.update(result)
        return res_dict
