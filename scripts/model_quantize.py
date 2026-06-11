import os
import gc
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-1.5B"
GPTQ_SAVE_PATH = "./qwen-1.5b-gptq-4bit"
AWQ_SAVE_PATH = "./qwen-1.5b-awq-4bit"
CALIBRATION_SAMPLES = 128

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# SHARED DATASET LOADER

def get_calibration_dataset(tokenizer, format_for_gptq=False):
    print("Downloading WikiText-2 calibration dataset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    valid_texts = [text for text in dataset["text"] if text.strip() != '' and len(text.split(' ')) > 20]
    calib_texts = valid_texts[:CALIBRATION_SAMPLES]

    if not format_for_gptq:
        return calib_texts

    print("Tokenizing data for GPTQ...")
    gptq_examples = []
    for text in calib_texts:
        encodings = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        gptq_examples.append({"input_ids": encodings.input_ids[0], "attention_mask": encodings.attention_mask[0]})
    return gptq_examples


# GPTQ QUANTIZATION (Using Modern GPTQModel API)

def run_gptq_quantization():
    print("\n" + "="*40 + "\n STARTING GPTQ QUANTIZATION\n" + "="*40)
    from gptqmodel import GPTQModel, QuantizeConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True
    )

    print("Loading base model into memory via GPTQModel.load()...")
    # FIX: Initialize the model first
    model = GPTQModel.load(MODEL_ID, quantize_config)

    print("Preparing calibration data...")
    calibration_data = get_calibration_dataset(tokenizer, format_for_gptq=True)

    print("Applying WikiText-2 calibration data (this will take time)...")
    # FIX: Call .quantize() directly on the instantiated model object
    model.quantize(calibration_data)

    print(f"Saving GPTQ model to {GPTQ_SAVE_PATH}...")
    # FIX: Modern GPTQModel uses .save() instead of .save_quantized()
    model.save(GPTQ_SAVE_PATH)
    tokenizer.save_pretrained(GPTQ_SAVE_PATH)

    del model
    del tokenizer
    flush_memory()
    print("✅ GPTQ COMPLETE!")

# AWQ QUANTIZATION (Using AutoAWQ)

def run_awq_quantization():
    print("\n" + "="*40 + "\n STARTING AWQ QUANTIZATION\n" + "="*40)
    from awq import AutoAWQForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    print("Loading base model into GPU...")
    model = AutoAWQForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)

    calibration_data = get_calibration_dataset(tokenizer, format_for_gptq=False)
    print("Applying WikiText-2 calibration data...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)

    print(f"Saving AWQ model to {AWQ_SAVE_PATH}...")
    model.save_quantized(AWQ_SAVE_PATH)
    tokenizer.save_pretrained(AWQ_SAVE_PATH)

    del model
    del tokenizer
    flush_memory()
    print(" AWQ COMPLETE")

if __name__ == "__main__":
    run_gptq_quantization()
    run_awq_quantization()
    print("\ ALL QUANTIZATION TASKS FINISHED SUCCESSFULLY")