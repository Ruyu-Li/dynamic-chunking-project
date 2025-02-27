# Environment Require
# !pip install llama-index-core
# !pip install llama-index-llms-openai
# !pip install llama-index-llms-replicate
# !pip install llama-index-embeddings-huggingface
# !pip install llama-index-embeddings-instructor
# !pip install llama-index-llms-gemini
# !pip install -U llama-index-vector-stores-chroma
# !pip install -U llama-index-readers-file
# !pip install bitsandbytes
# !pip install llama-index-readers-web
# !pip install llama-index-llms-huggingface
# !pip install llama-index-llms-huggingface-api
# !pip install "transformers[torch]" "huggingface_hub[inference]"
# !pip install llama-index-embeddings-gemini
# !pip install llama-index 'google-generativeai>=0.3.0' matplotlib
# !pip install transformers huggingface_hub

from llama_index.core import Settings
import os
from typing import List, Optional
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from huggingface_hub import login
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
import pickle
from llama_index.core import Settings

# Settings.llm = llm
# Setting chunk size and overlap, 128 and 20 are default
Settings.chunk_size = 128
Settings.chunk_overlap = 20

# SEE: https://huggingface.co/docs/hub/security-tokens
# We just need a token with read permissions for this demo
HF_TOKEN: Optional[str] = os.getenv("YOUR_HF_TOKEN")
# NOTE: None default will fall back on Hugging Face's token storage
# when this token gets used within HuggingFaceInferenceAPI

# 使用 API token 进行登录
login(token="YOUR_HF_TOKEN")


embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.embed_model = embed_model


# 文件路径
collection_path = "/content/drive/MyDrive/Narrativeqa/1wq_1.5wa_deduplication/context.tsv"

# 加载文档集合文件
collection_df = pd.read_csv(collection_path, sep='\t', header=None, names=["id", "context"])
collection_df["context"] = collection_df["context"].astype(str)

# 将文档转换为 Document 对象格式
documents = [Document(text=row["context"]) for _, row in collection_df.iterrows()]

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 保存索引文件
save_path = "/content/drive/MyDrive/Narrativeqa/1wq_1.5wa_deduplication/vector_index_256.pkl"
with open(save_path, "wb") as f:
    pickle.dump(index, f)