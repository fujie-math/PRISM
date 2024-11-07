import os
import sys
import torch
import json
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

model_type = "7b"
model_family = "llamachat"
print(f"{model_family}{model_type}")

model_path = "meta-llama/Llama-2-7b-chat-hf"  
#model_path = "meta-llama/Llama-2-13b-chat-hf"  
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


data_type_1 = ["animal_class", "cities", "element_symb", "facts", "inventors", "sp_en_trans"]   
data_type_2 = ["", "_neg", "_conj", "_disj"]
prompt = "Does the statement '{s}' accurately reflect the truth?"
storage_path = "../hd_data_prompt/logic/llama2chat7b/"
#storage_path = "../hd_data_prompt/logic/llama2chat13b/"


for dt1 in data_type_1:
    for dt2 in data_type_2:
        dt = f"{dt1}{dt2}"
        with open(f"../raw_data/logic/{dt}.json", "r", encoding='utf-8') as f:
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

