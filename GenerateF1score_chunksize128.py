import logging
import sys
import pandas as pd
import time
import pickle
import re
import numpy as np
import openai
import torch
import os
import nest_asyncio
from llama_index.core import Settings
from transformers import BitsAndBytesConfig
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI 
from typing import List, Optional
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from huggingface_hub import login
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response.notebook_utils import display_response
from llama_index.core import PromptTemplate
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

Settings.chunk_size = 128
Settings.chunk_overlap = 20

# **设置 OpenAI API Key**
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model="gpt-4o-mini",temperture = 0.2)
Settings.llm = llm  # 设置 LLM 到 LlamaIndex 的 `Settings` 配置


# SEE: https://huggingface.co/docs/hub/security-tokens
# We just need a token with read permissions for this demo
HF_TOKEN: Optional[str] = os.getenv("YOUR_HUGGINGFACE_TOKEN")
# NOTE: None default will fall back on Hugging Face's token storage
# when this token gets used within HuggingFaceInferenceAPI

# 使用 API token 进行登录
login(token="YOUR_HUGGINGFACE_TOKEN")


# Settings.llm = llm
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.embed_model = embed_model

# 查询文件路径
queries_path = "/media/volume/sigir/Narrativeqa/Narrativeqa/1wq_1.5wa_deduplication/remaining_queries.tsv"

# 加载查询文件
queries_df = pd.read_csv(queries_path, sep='\t', header=None, names=["id", "query"])
eval_questions = queries_df["query"].tolist()  # 提取查询文本列表



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
nest_asyncio.apply()

with open('/media/volume/sigir/Narrativeqa/Narrativeqa/1wq_1.5wa_deduplication/vector_index_128.pkl', 'rb') as f: # 加载vector_index
    vector_index = pickle.load(f)

# 加载 query300.tsv 文件

query_path = "/media/volume/sigir/Narrativeqa/Narrativeqa/1wq_1.5wa_deduplication/remaining_queries.tsv" # 加载query
query_data = pd.read_csv(query_path, sep='\t', skiprows=1, names=["id", "query", "answer"])

# 提取问题和答案
eval_questions = query_data["query"].tolist()
query_data["answer"] = query_data["answer"].fillna("No Answer Provided")
# eval_answers = query_data["answer"].apply(lambda x: x.split("; ")).tolist()
eval_answers = query_data["answer"].apply(lambda x: [gt.strip() for gt in x.split(";") if gt.strip()]).tolist()
print("Queries loaded successfully!")

# 上面都不需要动

prompt_template_str = f"""
You are a concise answer generator. Use the following context to directly answer the question.
Avoid providing any additional explanations, filler text, or phrases like "The answer is."
If the context doesn't contain enough information, simply reply "Not sure."

CONTEXT:
{{context_str}}

QUESTION:
{{query_str}}

Answer:
""".strip()

prompt_template = PromptTemplate(prompt_template_str)

# 错误计数器
error_count = 0
max_errors = 100

import re

def compute_f1(predicted, ground_truth_input):
    # 确保 predicted 是字符串
    if not isinstance(predicted, str):
        predicted = str(predicted)

    # 将 ground_truth_input 统一拆分成 ground_truths 列表
    if isinstance(ground_truth_input, str):
        # 按分号拆分字符串
        ground_truths = [gt.strip() for gt in ground_truth_input.split(';') if gt.strip()]
    elif isinstance(ground_truth_input, (list, tuple)):
        # 如果已经是列表，直接使用
        ground_truths = [str(gt).strip() for gt in ground_truth_input if str(gt).strip()]
    else:
        # 其他情况，转换为字符串并放入列表
        ground_truths = [str(ground_truth_input).strip()]

    # 正则化函数
    def normalize_answer(s):
        """Lower text and remove punctuation, articles, and extra whitespace."""
        s = re.sub(r'\b(a|an|the)\b', ' ', s)  # 去除冠词
        s = re.sub(r'[^a-zA-Z0-9\s]', '', s)  # 去除标点符号
        return ' '.join(s.lower().split())  # 转换为小写并去除多余空格

    # 计算单个 predicted 和 ground truth 的 precision, recall, F1
    def p_r_f1_single(pred, truth):
        """Compute precision, recall, and F1 for a single pair of predicted & ground truth."""
        pred_tokens = normalize_answer(pred).split()  # 正则化 predicted answer
        truth_tokens = normalize_answer(truth).split()  # 正则化 ground truth
        common = set(pred_tokens) & set(truth_tokens)  # 计算共同词汇
        if not common:
            return 0.0, 0.0, 0.0  # 如果没有共同词汇，F1 为 0
        precision = len(common) / len(pred_tokens)  # 计算 precision
        recall = len(common) / len(truth_tokens)  # 计算 recall
        if precision + recall == 0:
            return 0.0, 0.0, 0.0  # 防止除零错误
        f1 = 2 * (precision * recall) / (precision + recall)  # 计算 F1
        return precision, recall, f1

    # 计算最佳 F1
    best_p, best_r, best_f1 = 0.0, 0.0, 0.0
    for truth in ground_truths:
        p, r, f = p_r_f1_single(predicted, truth)
        if f > best_f1:
            best_p = p
            best_r = r
            best_f1 = f

    return best_p, best_r, best_f1

# Accuracy 计算函数
def compute_accuracy(predicted: str, ground_truths: list[str]) -> float:


    if not isinstance(predicted, str):
        predicted = str(predicted)
    for truth in ground_truths:
        if truth not in predicted:
            return 0.0
    return 1.0

# 计算答案的余弦相似度
def compute_embedding_similarity(predicted: str, ground_truths: list[str]) -> float:


    if not isinstance(predicted, str):
        predicted = str(predicted)
    if not ground_truths:
        return 0.0

    # 获取 predicted 的 embedding
    pred_emb = embed_model.get_text_embedding(predicted)
    pred_norm = np.linalg.norm(pred_emb) + 1e-10

    best_sim = 0.0
    for truth in ground_truths:
        gt_emb = embed_model.get_text_embedding(truth)
        gt_norm = np.linalg.norm(gt_emb) + 1e-10
        cos_sim = float(np.dot(pred_emb, gt_emb) / (pred_norm * gt_norm))
        if cos_sim > best_sim:
            best_sim = cos_sim

    return best_sim

# 保存结果到文件
def save_results(content_file, score_file, question_index, similarity_top_k, question,
                 full_prompt, response_vector, elapsed_time,
                 precision, recall, f1, accuracy, embed_sim, generated_answer):
    content_file.write(f"Question {question_index + 1}, similarity_top_k = {similarity_top_k}:\n")
    content_file.write(f"  Input Query: {question}\n")
    content_file.write(f"  Full Prompt: {full_prompt}\n")
    content_file.write(f"  Response Time: {elapsed_time:.2f}s\n")
    content_file.write(f"  Generated Answer: {generated_answer}\n")

    for j, node in enumerate(response_vector.source_nodes, start=1):
        text = node.node.get_text()
        content_file.write(f"  Fragment ID: Fragment_{j}\n")
        content_file.write(f"    Content: {text}\n")

    score_file.write(f"Question {question_index + 1}, similarity_top_k = {similarity_top_k}:\n")
    score_file.write(f"  Response Time: {elapsed_time:.2f}s\n")
    score_file.write(f"  Accuracy: {accuracy:.2f}\n")
    score_file.write(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}\n")
    score_file.write(f"  Embedding Sim: {embed_sim:.4f}\n\n")

# 评估函数
def evaluate_response_time_and_accuracy(similarity_top_k, eval_questions, eval_answers,
                                        content_file_path, score_file_path,
                                        max_retries=2, retry_delay=20):
    global error_count  # 使用全局错误计数器
    detailed_results = []


    query_engine = vector_index.as_query_engine(similarity_top_k=similarity_top_k)

    with open(content_file_path, 'w') as content_file, open(score_file_path, 'w') as score_file:
        for i, (question, ground_truths) in enumerate(zip(eval_questions, eval_answers)):
            retries = 0
            while retries <= max_retries:
                try:
                    start_time = time.time()
                    response_vector = query_engine.query(question)
                    elapsed_time = time.time() - start_time

                    # 拼接 Prompt
                    context_str = "\n".join([
                        node.node.get_text() for node in response_vector.source_nodes
                    ])
                    full_prompt = prompt_template_str.format(
                        context_str=context_str, query_str=question
                    )

                    # 获取生成的答案
                    generated_answer = str(response_vector) if response_vector else "No Answer Generated"

                    # 计算各种指标
                    accuracy = compute_accuracy(generated_answer, ground_truths)
                    precision, recall, f1 = compute_f1(generated_answer, ground_truths)
                    embed_sim = compute_embedding_similarity(generated_answer, ground_truths)

                    # 写入结果
                    save_results(
                        content_file, score_file, i, similarity_top_k, question, full_prompt,
                        response_vector, elapsed_time,
                        precision, recall, f1, accuracy, embed_sim, generated_answer
                    )

                    # 保存到列表
                    result = {
                        "question_index": i,
                        "question": question,
                        "response_time": elapsed_time,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "embedding_sim": embed_sim,
                        "generated_answer": generated_answer,
                        "ground_truths": ground_truths
                    }
                    detailed_results.append(result)

                    break  # 当前问题完成，退出重试循环

                except Exception as e:
                    error_count += 1
                    error_message = str(e)
                    print(f"Error processing question {i + 1}: {error_message}")
                    print(f"Total errors encountered: {error_count}/{max_errors}")

                    # 写入错误日志
                    with open("/media/volume/sigir/Narrativeqa/Narrativeqa/cs256_qr300_chatgpt/error_log.txt", 'a') as error_log_file:
                        error_log_file.write(f"Question {i + 1} failed with exception: {error_message}\n")

                    if error_count >= max_errors:
                        print("Error limit reached. Terminating the program.")
                        exit(1)

                    # 429 错误重试
                    if "429" in error_message:
                        retries += 1
                        if retries <= max_retries:
                            print(f"429 Error encountered. Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                            time.sleep(retry_delay)
                        else:
                            break
                    else:
                        break
    return detailed_results


results_list = []
for similarity_top_k in [5,6,7]: # 这里修改similarity_top_k
    print(f"Evaluating similarity_top_k: {similarity_top_k}")

    content_file_path = f"/media/volume/sigir/Narrativeqa/Narrativeqa/cs128_qr500_Chatgpt/content_results_top_k_{similarity_top_k}_.txt" #内容输出
    score_file_path = f"/media/volume/sigir/Narrativeqa/Narrativeqa/cs128_qr500_Chatgpt/evaluation_scores_top_k_{similarity_top_k}.txt" # 结果输出（F1）

    detailed_results = evaluate_response_time_and_accuracy(
        similarity_top_k, eval_questions, eval_answers,
        content_file_path, score_file_path
    )
    results_list.extend(detailed_results)

print("Evaluation complete. Collected results:", len(results_list))
