"""
Model configurations and constants for LLM Oracle
"""

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
MODEL_EMBED_SMALL = 'text-embedding-3-small'
MODEL_EMBED_LARGE = 'text-embedding-3-large'
MODEL_EMBED_GEMINI = 'text-embedding-004'

# google models
MODEL_GEMINI_15_PRO = 'gemini-1.5-pro'
MODEL_GEMINI_15_FLASH = 'gemini-1.5-flash'
MODEL_GEMINI_25_FLASH = 'gemini-2.5-flash'
MODEL_GEMINI_2_FLASH = 'gemini-2.0-flash-exp'
MODEL_GEMINI_1_PRO = 'gemini-1.0-pro'
MODEL_EMBED_GOOGLE = 'text-embedding-004'

# azure deployments
MODEL_PHI_3_MINI = 'phi-3-mini'
MODEL_PHI_3_5_MINI = 'phi-3.5-mini'
MODEL_PHI_3_SMALL = 'phi-3-small'
MODEL_PHI_3_MEDIUM = 'phi-3-medium'

# deepinfra deployments
DEEP_INFRA_BASE_URL = 'https://api.deepinfra.com/v1/openai'
MODEL_LLAMA_3_8B = 'llama-3-8B'
MODEL_LLAMA_3_70B = 'llama-3-70B'
MODEL_MIXTRAL_8X7B = 'mixtral-8x7B'

DEEP_INFRA_MAP = {
    MODEL_LLAMA_3_8B: 'meta-llama/Meta-Llama-3-8B-Instruct',
    MODEL_LLAMA_3_70B: 'meta-llama/Meta-Llama-3-70B-Instruct',
    MODEL_MIXTRAL_8X7B: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
}

# deepseek deployments
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1'
MODEL_DEEPSEEK_CHAT = 'deepseek-chat'
MODEL_DEEPSEEK_CODE = 'deepseek-code'
MODEL_DEEPSEEK_TOOLCHAT = 'deepseek-toolchat'

# No need for model mapping as DeepSeek uses these names directly
DEEPSEEK_MODELS = [MODEL_DEEPSEEK_CHAT, MODEL_DEEPSEEK_CODE, MODEL_DEEPSEEK_TOOLCHAT]

# other configs
MAX_COMPLETION_TOKENS = 1000

# Model lists for different providers
openai_model_list = [
    MODEL_GPT4_TURBO, MODEL_GPT4o, MODEL_GPT4o_MINI, MODEL_GPT_4_1,
    MODEL_GPT_5, MODEL_GPT_5_MINI, MODEL_GPT_o1, MODEL_GPT_o1_MINI,
    MODEL_GPT_o3, MODEL_GPT_o3_MINI, MODEL_EMBED_SMALL, MODEL_EMBED_LARGE
]

google_model_list = [
    MODEL_GEMINI_15_PRO, MODEL_GEMINI_15_FLASH, MODEL_GEMINI_2_FLASH, MODEL_GEMINI_25_FLASH,
    MODEL_GEMINI_1_PRO, MODEL_EMBED_GEMINI
]

azure_model_list = [
    MODEL_PHI_3_MINI, MODEL_PHI_3_5_MINI,
    MODEL_PHI_3_SMALL, MODEL_PHI_3_MEDIUM
]

deepinfra_model_list = [
    MODEL_LLAMA_3_8B, MODEL_LLAMA_3_70B, MODEL_MIXTRAL_8X7B
]

deepseek_model_list = DEEPSEEK_MODELS

# Combine all model lists
all_model_list = []
for l in [openai_model_list, google_model_list, azure_model_list, deepinfra_model_list, deepseek_model_list]:
    all_model_list += l

# Models that don't support top_p parameter
MODELS_WITHOUT_TOP_P = [
    MODEL_GPT_o1, MODEL_GPT_o1_MINI,
    MODEL_GPT_o3, MODEL_GPT_o3_MINI,
    MODEL_GPT_5, MODEL_GPT_5_MINI
]

# Models that require the default temperature value (API rejects overrides)
MODELS_WITH_FIXED_TEMPERATURE = [
    MODEL_GPT_5, MODEL_GPT_5_MINI
]
