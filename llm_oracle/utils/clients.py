"""
Client handlers for different model providers
"""

import time
from google.generativeai import types  # 修复 types 导入
from azure.ai.inference.models import SystemMessage, UserMessage  # 修复 SystemMessage 和 UserMessage 导入

from ..config.models import (
    MODELS_WITHOUT_TOP_P,
    MAX_COMPLETION_TOKENS,
    DEEP_INFRA_MAP
)
from .response_handlers import response_failure, response_failure_embed

def openai_chat_query(
        client,
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
    """
    Handle OpenAI chat completion query
    """
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
            }
            
            if model not in MODELS_WITHOUT_TOP_P:
                completion_params["top_p"] = top_p
            
            completion = client.chat.completions.create(**completion_params)

            response_result = ""
            if completion.choices[0].message:
                response_result += completion.choices[0].message.content
            
            if not query_key:
                query_key = prompt_user
            return {"query": query_key, "answer": response_result}

        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                return response_failure(query_key, model, e)

def openai_embed_query(client, model, list_of_text, dimensions, original_key=False, max_retries=3, retry_delay=2) -> dict:
    """
    Handle OpenAI embedding query
    """
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text_clean = [text.replace("\n", " ") for text in list_of_text]

    retry_count = 0
    while retry_count <= max_retries:
        try:
            data = client.embeddings.create(
                input=list_of_text_clean,
                model=model,
                dimensions=dimensions,
                timeout=30
            ).data
            break
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                return response_failure_embed(list_of_text, model, e)
    
    list_of_keys = list_of_text if original_key else list_of_text_clean
    res = {list_of_keys[i]: data[i].embedding for i in range(len(list_of_keys))}
    return res

def google_embed_query(client, model, list_of_text, dimensions, original_key=False) -> dict:
    """
    Handle Google embedding query
    """
    embed_config = types.EmbedContentConfig(output_dimensionality=dimensions)
    data = client.models.embed_content(
        model=model,
        contents=list_of_text,
        config=embed_config
    )
    embds = [e.values for e in data.embeddings]
    res = {list_of_text[i]: embds[i] for i in range(len(list_of_text))}
    return res

def google_chat_query(client, model, prompt_sys, prompt_user, temp, top_p, query_key=None) -> dict:
    """
    Handle Google chat completion query
    """
    gen_config = {"temperature": temp}
    try:
        response = client.generate_content(prompt_user, generation_config=gen_config)
        if not query_key:
            query_key = prompt_user
        res = {"query": query_key, "answer": response.text}
        return res
    except Exception as e:
        return response_failure(prompt_user, model, e)
    
def azure_chat_query(client, model, prompt_sys, prompt_user, temp, top_p, query_key=None) -> dict:
    """
    Handle Azure chat completion query
    """
    sys_message = SystemMessage(content=prompt_sys)
    usr_message = UserMessage(content=prompt_user)
    try:
        response = client.complete(
            messages=[sys_message, usr_message],
            temperature=temp,
        )
        response_text = response['choices'][0]['message']['content']
        if not query_key:
            query_key = prompt_user
        res = {"query": query_key, "answer": response_text}
        return res
    except Exception as e:
        return response_failure(prompt_user, model, e)

def deepseek_chat_query(
        client,
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
    """
    Handle DeepSeek chat completion query
    Based on https://api-docs.deepseek.com/zh-cn/api/chat
    """
    retry_count = 0
    while retry_count <= max_retries:
        try:
            messages = []
            if prompt_sys:
                messages.append({"role": "system", "content": prompt_sys})
            messages.append({"role": "user", "content": prompt_user})
            
            completion_params = {
                "model": model,
                "messages": messages,
                "temperature": temp,
                "top_p": top_p,
                "max_tokens": max_completion_tokens,
                "stream": False,
                # 可选参数
                # "presence_penalty": 0,
                # "frequency_penalty": 0,
                # "stop": None,
                # "tools": None,
                # "tool_choice": None
            }
            
            completion = client.chat.completions.create(**completion_params)

            response_result = ""
            if completion.choices[0].message:
                response_result += completion.choices[0].message.content
            
            if not query_key:
                query_key = prompt_user
                
            # 包含完整的usage信息
            usage = {}
            if hasattr(completion, 'usage'):
                usage = completion.usage._asdict() if hasattr(completion.usage, '_asdict') else vars(completion.usage)
            elif isinstance(completion, dict) and 'usage' in completion:
                usage = completion['usage']
            
            return {
                "query": query_key,
                "answer": response_result,
                "usage": usage
            }

        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                return response_failure(query_key, model, e)
