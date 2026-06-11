#!/bin/bash

echo "=================================================="
echo "🧪 Running MMLU & GSM8k Evaluation for AWQ..."
echo "=================================================="
lm_eval --model vllm \
  --model_args pretrained="/content/qwen_model_local/qwen-1.5b-awq-4bit",quantization="awq",gpu_memory_utilization=0.7,max_model_len=1024,max_num_seqs=256 \
  --tasks mmlu_abstract_algebra,mmlu_anatomy,gsm8k \
  --num_fewshot 5 \
  --limit 50

echo "=================================================="
echo "🧪 Running MMLU & GSM8k Evaluation for GPTQ..."
echo "=================================================="
lm_eval --model vllm \
  --model_args pretrained="/content/qwen_gptq_local/qwen-1.5b-gptq-4bit",quantization="gptq",gpu_memory_utilization=0.7,max_model_len=1024,max_num_seqs=256 \
  --tasks mmlu_abstract_algebra,mmlu_anatomy,gsm8k \
  --num_fewshot 5 \
  --limit 50

echo "=================================================="
echo "📖 Running Perplexity (WikiText) Evaluation for AWQ..."
echo "=================================================="
lm_eval --model vllm \
  --model_args pretrained="/content/qwen_model_local/qwen-1.5b-awq-4bit",quantization="awq",gpu_memory_utilization=0.7,max_model_len=1024,max_num_seqs=256 \
  --tasks wikitext \
  --limit 50

echo "=================================================="
echo "📖 Running Perplexity (WikiText) Evaluation for GPTQ..."
echo "=================================================="
lm_eval --model vllm \
  --model_args pretrained="/content/qwen_gptq_local/qwen-1.5b-gptq-4bit",quantization="gptq",gpu_memory_utilization=0.7,max_model_len=1024,max_num_seqs=256 \
  --tasks wikitext \
  --limit 50