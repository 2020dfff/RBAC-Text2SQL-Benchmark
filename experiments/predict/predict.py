import json
import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from typing import Any, Dict, List, Optional

from experiments.data_process.data_utils import extract_sql_prompt_dataset, extract_sql_role_prompt_dataset
from experiments.llm_base.chat_model import ChatModel
from tqdm import tqdm


def prepare_dataset(
    predict_file_path: Optional[str] = None,
) -> List[Dict]:
    with open(predict_file_path, "r") as fp:
        data = json.load(fp)
    
    # Check if the data contains role information
    if data and isinstance(data[0], dict) and "role" in data[0]:
        predict_data = [extract_sql_role_prompt_dataset(item) for item in data]
    else:
        predict_data = [extract_sql_prompt_dataset(item) for item in data]
    return predict_data


def inference(model: ChatModel, predict_data: List[Dict], **input_kwargs):
    res = []
    # test
    # for item in predict_data[:20]:
    for item in tqdm(predict_data, desc="Inference Progress", unit="item"):
        # print(f"item[input] \n{item['input']}")
        response, _ = model.chat(query=item["input"], history=[], **input_kwargs)
        res.append(response)
    return res

    # batch_size = input_kwargs.get("batch_size", 8)  # 默认batch size为8，可通过参数传入
    # num_samples = len(predict_data)
    # for start_idx in tqdm(range(0, num_samples, batch_size), desc="Inference Progress", unit="batch"):
    #     batch = predict_data[start_idx:start_idx+batch_size]
    #     batch_inputs = [item["input"] for item in batch]
    #     # print input of each batch
    #     for inp in batch_inputs:
    #         print(f"item[input] \n{inp}")
    #     # assume model.chat supports batch input, otherwise need to implement batch_chat in ChatModel
    #     if hasattr(model, "batch_chat"):
    #         responses = model.batch_chat(queries=batch_inputs, histories=[[]]*len(batch), **input_kwargs)
    #     else:
    #         # fallback: single inference
    #         print("Warning: model does not support batch_chat, falling back to single inference.")
    #         responses = []
    #         for inp in batch_inputs:
    #             response, _ = model.chat(query=inp, history=[], **input_kwargs)
    #             responses.append(response)
    #     res.extend(responses)
    # return res


def predict(model: ChatModel):
    args = model.data_args
    ## predict file can be give by param --predicted_input_filename ,output_file can be gived by param predicted_out_filename
    predict_data = prepare_dataset(args.predicted_input_filename)
    # predict_data = predict_data[:10]  # only take first ten records for test, comment this line for full evaluation
    result = inference(model, predict_data)
    # batch_size = getattr(args, "batch_size", 8)
    # result = inference(model, predict_data, batch_size=batch_size)

    with open(args.predicted_out_filename, "w") as f:
        for p in result:
            try:
                f.write(p.replace("\n", " ") + "\n")
            except:
                f.write("Invalid Output!\n")


if __name__ == "__main__":
    model = ChatModel()
    predict(model)
