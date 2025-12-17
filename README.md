day5-rag
day5-rag

1. Upgrade pip
pip install --upgrade pip setuptools wheel

2. Install vLLM (official method)
pip install vllm==0.11.0

3. Install LlamaIndex with conflict-free versions
pip install llama-index-core
llama-index-llms-vllm
llama-index-embeddings-huggingface
--no-cache-dir
