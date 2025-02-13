import os
import sys
import torch
import json
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_path = "meta-llama/Llama-2-7b-chat-hf"
  
model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map='auto')
tokenizer = LlamaTokenizer.from_pretrained(model_path)


def get_hd(q):
    
    text = q.strip()
    id1 = tokenizer.encode(text)
    hd = model(torch.tensor([id1]).to(model.device), output_hidden_states=True).hidden_states    
    hds = hd[-1][0][-1].clone().detach()

    return hds.tolist()


data_type = ["animals", "cities", "companies","elements", "facts", "inventions"]   
prompt = "Here is a statement: {s}\nIs the above statement correct?"
storage_path = "./"


for dt in data_type:
    with open(f"../../raw_data/true/{dt}.json", "r", encoding='utf-8') as f:
        data = json.load(f)

    results_last = []
    for stc in tqdm(data):
        s = stc["statement"]
        formatted_prompt = prompt.format(s=s)
        hdl = get_hd(formatted_prompt)
        
        results_last.append({
            "hd": hdl,
            "label": stc["label"]
        })

    with open(f"{storage_path}/{dt}_last_token.json", "w+") as f:
        json.dump(results_last, f)

