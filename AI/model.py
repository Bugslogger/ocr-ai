import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


def download_ocr():
    model_name = "baidu/Unlimited-OCR"

    ocr_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ocr_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    # ocr_model = ocr_model.eval().cuda()
    print("Unlimited-OCR loaded.")
    return {"ocr_tokenizer": ocr_tokenizer, "ocr_model": ocr_model}

def download_qwen():
    LLM_ID = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {LLM_ID}...")
    Qtokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    Qmodel = AutoModelForCausalLM.from_pretrained(
        LLM_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Qmodel.eval()
    print("Qwen loaded.")
    return {"Qtokenizer": Qtokenizer, "Qmodel": Qmodel}
    
    
