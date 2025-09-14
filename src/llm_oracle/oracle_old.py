import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

# 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# import google.generativeai as genai # old
from google import genai # new
# from google.genai import types
# from azure.ai.inference import ChatCompletionsClient
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.inference.models import SystemMessage, UserMessage

# openai apis
MODEL_GPT4_TURBO = 'gpt-4-turbo'
MODEL_GPT4o = 'gpt-4o'
MODEL_GPT4o_MINI = 'gpt-4o-mini'
MODEL_GPT_4_1 = 'gpt-4.1'
MODEL_GPT_5 = 'gpt-5'
MODEL_GPT_5_MINI = 'gpt-5-mini'
MODEL_GPT_o1 = 'o1'
MODEL_GPT_o1_MINI = 'o1-mini'
MODEL_GPT_o3 = 'o3'
MODEL_GPT_o3_MINI = 'o3-mini'
# MODEL_GPT4 = 'gpt-4'                  # out-dated, should not use
# MODEL_GPT3_5 = 'gpt-3.5-turbo-0125'   # out-dated, should not use
MODEL_EMBED_SMALL = 'text-embedding-3-small'
MODEL_EMBED_LARGE = 'text-embedding-3-large'
MODEL_EMBED_GEMINI = 'text-embedding-004'
# google models
MODEL_GEMINI_15_PRO = 'gemini-1.5-pro'
MODEL_GEMINI_15_FLASH = 'gemini-1.5-flash'
MODEL_GEMINI_1_PRO = 'gemini-1.0-pro'
MODEL_EMBED_GOOGLE = 'text-embedding-004'
# azure deployments (for phi-family models you need to specify the endpoint url by yourself)
MODEL_PHI_3_MINI = 'phi-3-mini'
MODEL_PHI_3_5_MINI = 'phi-3.5-mini'
MODEL_PHI_3_SMALL = 'phi-3-small'
MODEL_PHI_3_MEDIUM = 'phi-3-medium'
# deepinfra deployments
DEEP_INFRA_BASE_URL = 'https://api.deepinfra.com/v1/openai'
MODEL_LLAMA_3_8B = 'llama-3-8B'
MODEL_LLAMA_3_70B = 'llama-3-70B'
MODEL_MIXTRAL_8X7B = 'mixtral-8x7B'
DEEP_INFRA_MAP = {MODEL_LLAMA_3_8B: 'meta-llama/Meta-Llama-3-8B-Instruct',
                  MODEL_LLAMA_3_70B: 'meta-llama/Meta-Llama-3-70B-Instruct',
                  MODEL_MIXTRAL_8X7B: 'mistralai/Mixtral-8x7B-Instruct-v0.1',}

# other configs
MAX_COMPLETION_TOKENS = 1000

openai_model_list = [MODEL_GPT4_TURBO, MODEL_GPT4o, MODEL_GPT4o_MINI,  MODEL_GPT_4_1, MODEL_GPT_5, MODEL_GPT_5_MINI,
                     MODEL_GPT_o1, MODEL_GPT_o1_MINI, MODEL_GPT_o3, MODEL_GPT_o3_MINI,
                     MODEL_EMBED_SMALL, MODEL_EMBED_LARGE]
google_model_list = [MODEL_GEMINI_15_PRO, MODEL_GEMINI_15_FLASH, MODEL_GEMINI_1_PRO, MODEL_EMBED_GEMINI]
azure_model_list = [MODEL_PHI_3_MINI, MODEL_PHI_3_5_MINI, MODEL_PHI_3_SMALL, MODEL_PHI_3_MEDIUM]
deepinfra_model_list = [MODEL_LLAMA_3_8B, MODEL_LLAMA_3_70B, MODEL_MIXTRAL_8X7B]
all_model_list = []
for l in [openai_model_list, google_model_list, azure_model_list, deepinfra_model_list]:
    all_model_list += l

MODELS_WITHOUT_TOP_P = [MODEL_GPT_o1, MODEL_GPT_o1_MINI, MODEL_GPT_o3, MODEL_GPT_o3_MINI, MODEL_GPT_5, MODEL_GPT_5_MINI]

def response_failure(prompt_user, model, e,):
    print(f"err: The following error occurred when querying {prompt_user} through {model}:")
    print(e)
    return {"query": prompt_user, "answer": "QUERY_FAILED"}

def response_failure_embed(list_of_text, model, e):
    print(f"err: The following error occurred when querying the following list of text through {model}:")
    print(list_of_text)
    print(e)
    return {"query": list_of_text, "answer": ["QUERY_FAILED"]*len(list_of_text)}

def openai_chat_query(
        # single query
        client, # openai client object;
        model,
        prompt_sys,
        prompt_user,
        temp,
        top_p,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        query_key=None,
        max_retries=1,
        retry_delay=2,
        quiet=True,
    ) -> dict:

        retry_count = 0
        while retry_count <= max_retries:
            try:
                
                completion_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt_sys},
                        {"role": "user", "content": prompt_user},
                    ],
                    "stream": False,
                    "temperature": temp,
                    "max_completion_tokens": max_completion_tokens,
                    # "timeout": 100,  # timeout
                }
                
                # Only add top_p for models that support it
                if model not in MODELS_WITHOUT_TOP_P:
                    completion_params["top_p"] = top_p
                
                completion = client.chat.completions.create(**completion_params)

                response_result = ""
                if completion.choices[0].message:
                    response_result += completion.choices[0].message.content
                
                if not query_key:
                    query_key = prompt_user 
                # if not quiet: logger.info(f"成功获取OpenAI响应: {query_key}")
                return {"query": query_key, "answer": response_result,}

            except Exception as e:  # Consider capturing a specific exception if possible
                retry_count += 1
                if retry_count <= max_retries:
                    # if not quiet: logger.warning(f"OpenAI请求失败 (第{retry_count}/{max_retries}次重试): {str(e)}, 等待{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # 指数退避
                else:
                    # if not quiet: logger.error(f"OpenAI请求失败 (已达到最大重试次数): {str(e)}")
                    return response_failure(query_key, model, e)

def openai_embed_query(client, model, list_of_text, dimensions, original_key=False, max_retries=3, retry_delay=2) -> dict:
    # small batch query
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text_clean = [text.replace("\n", " ") for text in list_of_text]

    retry_count = 0
    while retry_count <= max_retries:
        try:
            data = client.embeddings.create(
                input=list_of_text_clean,
                model=model,
                dimensions=dimensions,
                timeout=30  # 设置30秒超时
            ).data
            # logger.info(f"成功获取OpenAI嵌入向量: {len(list_of_text)}个文本")
            break
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                # logger.warning(f"OpenAI嵌入请求失败 (第{retry_count}/{max_retries}次重试): {str(e)}, 等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指数退避
            else:
                # logger.error(f"OpenAI嵌入请求失败 (已达到最大重试次数): {str(e)}")
                return response_failure_embed(list_of_text, model, e)
    
    list_of_keys = list_of_text if original_key else list_of_text_clean

    res = {list_of_keys[i]:data[i].embedding for i in range(len(list_of_keys))}
    return res

def google_embed_query(client, model, list_of_text, dimensions, original_key=False) -> dict:
    embed_config = types.EmbedContentConfig(output_dimensionality=dimensions)
    data = client.models.embed_content(model=model,
                                       contents=list_of_text,
                                       config=embed_config)
    embds = [e.values for e in data.embeddings]
    res = {list_of_text[i]:embds[i] for i in range(len(list_of_text))}
    return res

def google_chat_query(client, model, prompt_sys, prompt_user, temp, top_p, query_key=None) -> dict:
    # here model and prompt_sys are useless, just to align with the openai interface
    # TODO: support top_p
    gen_config = {"temperature": temp,}
    try:
        response = client.generate_content(prompt_user, generation_config=gen_config)
        if not query_key:
            query_key = prompt_user
        res =  {"query": query_key, "answer": response.text}
        return res
    except Exception as e:
        return response_failure(prompt_user, model, e)
    
def azure_chat_query(client, model, prompt_sys, prompt_user, temp, top_p, query_key=None) -> dict:
    # TODO: support top_p
    sys_message = SystemMessage(content=prompt_sys)
    usr_message = UserMessage(content=prompt_user)
    try:
        response = client.complete(messages=[sys_message, usr_message,], temperature=temp,)
        response_text = response['choices'][0]['message']['content']
        if not query_key:
            query_key = prompt_user
        res = {"query": query_key, "answer": response_text}
        return res
    except Exception as e:
        return response_failure(prompt_user, model, e)

class Oracle:
    def __init__(self, model, apikey=None, azure_end_point=''):
        assert model in all_model_list, f'err: model named {model} is not supported'
        self.model = model
        self.apikey = apikey
        # for openai models
        if model in openai_model_list:
            if apikey: openai.api_key = apikey
            self.client = OpenAI()
        elif model in deepinfra_model_list:
            self.client = OpenAI(api_key=apikey, base_url=DEEP_INFRA_BASE_URL)
        elif model in google_model_list:
            # genai.configure(api_key=apikey,)
            # self.client = genai.GenerativeModel(model)
            self.client = genai.Client(api_key=apikey)
        elif model in azure_model_list:
            azure_credential = AzureKeyCredential(apikey)
            self.client = ChatCompletionsClient(endpoint=azure_end_point, credential=azure_credential)
    
    # for chat completion
    def query(self, prompt_sys, prompt_user, temp=1.0, top_p=0.9, query_key=None, max_completion_tokens=MAX_COMPLETION_TOKENS):
        if self.model in openai_model_list:
            return openai_chat_query(self.client, self.model, prompt_sys, prompt_user, temp, top_p, max_completion_tokens, query_key)
        elif self.model in deepinfra_model_list:
            return openai_chat_query(self.client, DEEP_INFRA_MAP[self.model], prompt_sys, prompt_user, temp, top_p, max_completion_tokens, query_key)
        elif self.model in google_model_list:
            # prompt_sys not supported
            return google_chat_query(self.client, self.model, prompt_sys, prompt_user, temp, top_p, query_key)
        elif self.model in azure_model_list:
            return azure_chat_query(self.client, self.model, prompt_sys, prompt_user, temp, top_p, query_key)
    
    def query_all(self, prompt_sys, prompt_user_all, workers=12, temp=1.0, top_p=0.9, query_key_list=[], reorder=True, max_completion_tokens=MAX_COMPLETION_TOKENS, **kwargs):
        # prompt_user_all: [str,]

        if type(prompt_sys) == str:
            prompt_sys = [prompt_sys] * len(prompt_user_all)

        # collect procedure
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
            if prompt_sys: # refresh the system prompt
                self.client = genai.GenerativeModel(model_name=self.model, system_instruction=prompt_sys,)
        elif self.model in azure_model_list:
            single_query_fn = azure_chat_query
        
        # Handle reordering by creating indexed keys
        if reorder:
            if query_key_list:
                # Prepend index to existing query keys
                indexed_query_key_list = [f"{i:06d}_{key}" for i, key in enumerate(query_key_list)]
            else:
                # Create indexed keys based on enumeration
                indexed_query_key_list = [f"{i:06d}" for i in range(len(prompt_user_all))]
        else:
            indexed_query_key_list = query_key_list if query_key_list else []

        with ThreadPoolExecutor(max_workers=workers) as executor: # avg. 0.13s per item
            if indexed_query_key_list:
                if single_query_fn is openai_chat_query:
                    future_results = [
                        executor.submit(single_query_fn, self.client, model_name, prompt_sys[i], p, temp, top_p, max_completion_tokens, indexed_query_key_list[i])
                        for i, p in enumerate(prompt_user_all)
                    ]
                else:
                    future_results = [
                        executor.submit(single_query_fn, self.client, model_name, prompt_sys[i], p, temp, top_p, indexed_query_key_list[i])
                        for i, p in enumerate(prompt_user_all)
                    ]
            else:
                if single_query_fn is openai_chat_query:
                    future_results = [
                        executor.submit(single_query_fn, self.client, model_name, prompt_sys[i], p, temp, top_p, max_completion_tokens)
                        for i, p in enumerate(prompt_user_all)
                    ]
                else:
                    future_results = [
                        executor.submit(single_query_fn, self.client, model_name, prompt_sys[i], p, temp, top_p)
                        for i, p in enumerate(prompt_user_all)
                    ]
        
            for future in tqdm(as_completed(future_results), total=len(prompt_user_all), desc="Processing Items"):
                result = future.result()
                results.append(result)
                
        if reorder:
            # Sort results by the index in the query key
            results.sort(key=lambda x: int(x["query"].split("_")[0]))
            
            # Clean up the query keys by removing the index prefix
            for result in results:
                if "_" in result["query"]:
                    # Remove the index prefix (everything before and including the first underscore)
                    result["query"] = "_".join(result["query"].split("_")[1:])

        return results
    
    # for semantic embedding generation
    def encode(self, text, dim=1024):
        # single
        if self.model in openai_model_list:
            return openai_embed_query(self.client, self.model, [text], dim)
        elif self.model in google_model_list:
            return google_embed_query(self.client, self.model, [text], dim)
    
    def quick_encode(self, text, dim=1024): # directly return the embedding
        if self.model in openai_model_list:
            return openai_embed_query(self.client, self.model, [text], dim, original_key=True)[text]
        else:
            raise RuntimeError(f'unsupported error for model {self.model}')

    def encode_all(self, list_of_text, workers=12, dim=1024, chunk_size=100, original_key=False):
        # split the list into small sublists
        sublists = [list_of_text[i:i + chunk_size] for i in range(0, len(list_of_text), chunk_size)]
        
        results = []
        print(f"Total queries: {len(list_of_text)}, start collecting...")
        
        if self.model in openai_model_list:
            single_query_fn = openai_embed_query
        elif self.model in google_model_list:
            single_query_fn = google_embed_query
        else:
            raise RuntimeError(f'unsupported error for model {self.model}')

        with ThreadPoolExecutor(max_workers=workers) as executor: # avg. 0.13s per item
            future_results = [executor.submit(single_query_fn, self.client, self.model, p, dim, original_key) for p in sublists]
        
            for future in tqdm(as_completed(future_results), total=len(sublists), desc="Processing Items"):
                result = future.result()
                results.append(result)
        return results

    def collect_results(self, results):
        # where results is a list of k-v pairs, now we need to collect them together
        res_dict = {}
        for result in results:
            res_dict.update(result)
        return res_dict
