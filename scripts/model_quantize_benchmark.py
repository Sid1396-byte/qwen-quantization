import time
import numpy as np
import torch
import gc
from vllm import LLM, SamplingParams

# ==========================================
# CHANGE THESE FOR EACH RUN!
# (And remember to RESTART THE KERNEL between runs!)
# ==========================================

# ---- For the AWQ Run ----
MODEL_PATH = "/content/qwen_model_local/qwen-1.5b-awq-4bit"
QUANT_METHOD = "awq"

# ---- For the GPTQ Run ----
# MODEL_PATH = "/content/qwen_gptq_local/qwen-1.5b-gptq-4bit"
# QUANT_METHOD = "gptq"


def fair_benchmark(iterations=3):
    print(f"Loading model into vLLM Engine: {MODEL_PATH}")

    # EXPLICITLY pass the target engine
    llm = LLM(
        model=MODEL_PATH,
        quantization=QUANT_METHOD,
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        enforce_eager=False,
    )

    # 🕵️ DYNAMIC ENGINE INSPECTION
    # This reaches into vLLM's internal config to see what it actually compiled
    actual_quant = getattr(llm.llm_engine.model_config, 'quantization', 'unknown')

    print("\n" + "*"*55)
    print(f"🕵️ ENGINE INSPECTION:")
    print(f"Requested Method : {QUANT_METHOD}")
    print(f"Actual Backend   : {actual_quant.upper()}")

    if "marlin" in actual_quant.lower():
        print("SUCCESS: The ultra-fast Marlin Kernel is fully engaged!")
    else:
        print("WARNING: Marlin is NOT active. Falling back to standard kernels.")
    print("*"*55 + "\n")

    base_prompts = [
        "The future of artificial intelligence is",
        "To bake a perfect chocolate chip cookie, you must",
        "The history of the Roman Empire shows that",
        "In quantum physics, the concept of entanglement means"
    ]
    prompts = base_prompts * 8 # 32 total prompts

    # FORCED FAIRNESS
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.0,
        ignore_eos=True
    )

    print("Warming up engine (compiling CUDA graphs for batch size 32)...")
    _ = llm.generate(prompts, sampling_params, use_tqdm=False)

    print(f"Running benchmark for {iterations} iterations...\n")

    throughputs = []

    for i in range(iterations):
        # Sync CUDA to ensure accurate timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_generated_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        time_taken = end_time - start_time
        tok_per_sec = total_generated_tokens / time_taken
        throughputs.append(tok_per_sec)

        print(f"  Iteration {i+1}: {tok_per_sec:.2f} tokens/sec")

    avg_throughput = np.mean(throughputs)

    print("\n" + "="*45)
    print(f"THE ULTIMATE BENCHMARK ({QUANT_METHOD.upper()})")
    print("="*45)
    print(f"Batch Size (Prompts) : {len(prompts)}")
    print(f"Tokens per Prompt    : 256 (Forced)")
    print(f"Total Tokens per Run : {len(prompts) * 256}")
    print(f"Average Throughput   : {avg_throughput:.2f} tokens/second")
    print("="*45)

    # Attempt to free VRAM for the next run (Restarting the kernel is still safer)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

fair_benchmark()