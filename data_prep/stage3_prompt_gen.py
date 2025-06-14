import json
import numpy as np
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# from tools import get_json, output2triple, process_triple, check_response
# from qwen_gen import QwenGen
# from get_prompt import generate_rag_prompt

# 模拟函数
def get_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
def output2triple(s): return s # 模拟
def process_triple(s): return s # 模拟
def check_response(s): return True # 模拟
def generate_rag_prompt(query, context_item, type):
    return f"Context: {context_item['content']} -> {context_item['output']}\nQuery: {query}" # 模拟
class QwenGen:
    def __init__(self, port, temperature): self.port = port
    def response(self, prompt): return "模拟的Qwen输出" # 模拟



# 1. 全局设置与初始化
# ==============================================================================
# 知识库设置
TRAIN_DATA_PATH = '../data/train.json'
# 测试数据
TEST_DATA_PATH = '../data/test2.json'
# 模型设置
EMBEDDING_MODEL_PATH = "../models/bge-large-zh-v1.5"
# API服务器设置
VLLM_PORT = 35000
VLLM_TEMPERATURE = 0.1
# RAG 与 自洽性 (Self-Consistency) 设置
INTEGRATION_NUM = 5  # 每次检索的文档数量，也即投票的数量
THRESHOLD = 3      # 投票结果中，多数响应需要达到的最小票数
# 输出文件
OUTPUT_FILE_PATH = f'../data/output/stage3_rag_predictions_k{INTEGRATION_NUM}_t{THRESHOLD}.jsonl'


def retriever(corpus_embeddings_np: np.ndarray, query: str, texts: list[str], model: SentenceTransformer, top_k: int = 1) -> list[str]:
    """
    从文本库中检索与查询最相似的top_k个文本。

    Args:
        corpus_embeddings_np (np.ndarray): 整个知识库的预计算向量。
        query (str): 待查询的文本。
        texts (list[str]): 知识库中的原始文本列表。
        model (SentenceTransformer): 用于编码查询的向量化模型。
        top_k (int): 需要检索的最相似文本数量。

    Returns:
        list[str]: 检索到的top_k个文本列表。
    """
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)

    if query_embedding.is_cuda:
        query_embedding = query_embedding.cpu()

    query_embedding_np = query_embedding.numpy().reshape(1, -1)

    similarities = cosine_similarity(query_embedding_np, corpus_embeddings_np)[0]

    # 获取相似度最高的 top_k 个索引 (降序)
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    retrieved_texts = [texts[idx] for idx in top_k_indices]
    return retrieved_texts


def gen_output_with_consistency(item: dict, qwen_generator: QwenGen, texts: list, test2item: dict, corpus_embeddings_np: np.ndarray, model: SentenceTransformer) -> str:
    """
    使用RAG和自洽性策略生成稳定的输出。
    会一直重试直到多数响应达到阈值。

    Args:
        item (dict): 当前待测试的数据项。
        qwen_generator (QwenGen): Qwen模型的API客户端。
        ... (其他所需参数)

    Returns:
        str: 经过投票和自洽性检验的最终输出。
    """
    while True:
        # 1. 检索 K 个相关样本
        retrieved_contents = retriever(corpus_embeddings_np, item['content'], texts, model, top_k=INTEGRATION_NUM)
        
        all_responses = []
        # 2. 对每个样本独立生成 RAG prompt 并调用模型
        for retrieved_content in retrieved_contents:
            retrieved_item = test2item[retrieved_content]
            prompt = generate_rag_prompt(item['content'], retrieved_item, 'triple')
            result = qwen_generator.response(prompt)
            all_responses.append(result)

        # 3. 统计响应并进行投票
        if not all_responses:
            continue # 如果没有任何响应，则重试

        response_counts = Counter(all_responses)
        most_common_response, count = response_counts.most_common(1)[0]
        
        # 4. 检查是否满足一致性阈值
        if count >= THRESHOLD:
            print(f"    [Success] Found consistent response: '{most_common_response}' with {count}/{INTEGRATION_NUM} votes.")
            return most_common_response
        else:
            print(f"    [Retry] No consistent response found. Top response has {count}/{INTEGRATION_NUM} votes. Retrying...")


def main():
    """主执行函数"""
    # 1. 加载知识库和预计算向量
    print("Step 1: Loading knowledge base and embedding model...")
    train_data = get_json(TRAIN_DATA_PATH)
    texts = [item['content'] for item in train_data]
    
    # 创建一个从 content 到完整 item 的映射，方便快速查找
    content_to_item_map = {item['content']: item for item in train_data}
    for item in content_to_item_map.values():
        item['output'] = output2triple(item['output']) # 预处理知识库的输出格式

    print(f"Loading embedding model from: {EMBEDDING_MODEL_PATH}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    print("Model loaded. Computing embeddings for knowledge base...")
    
    corpus_embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    if corpus_embeddings.is_cuda:
        corpus_embeddings = corpus_embeddings.cpu()
    corpus_embeddings_np = corpus_embeddings.numpy()
    print("Knowledge base embeddings computed and ready.")

    # 2. 初始化LLM客户端
    print(f"\nStep 2: Initializing Qwen API client on port {VLLM_PORT}...")
    qwen_generator = QwenGen(port=VLLM_PORT, temperature=VLLM_TEMPERATURE)
    print("Qwen client ready.")

    # 3. 加载测试数据并执行预测流程
    print("\nStep 3: Starting prediction on test data...")
    test_data = get_json(TEST_DATA_PATH)
    
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        for i, item in enumerate(tqdm(test_data, desc="Processing test items")):
            print(f"\nProcessing item {i+1}/{len(test_data)}: {item['content']}")
            
            # 使用自洽性策略生成一个可靠的输出
            consistent_output = gen_output_with_consistency(item, qwen_generator, texts, content_to_item_map, corpus_embeddings_np, embedding_model)
            
            # 对输出进行后处理和格式校验
            processed_output = process_triple(consistent_output)
            
            # 循环直到格式校验通过 (这是您代码中的第二层鲁棒性保障)
            retry_count = 0
            while not check_response(processed_output):
                retry_count += 1
                print(f"    [Format Check Failed] Retrying generation... (Attempt {retry_count})")
                consistent_output = gen_output_with_consistency(item, qwen_generator, texts, content_to_item_map, corpus_embeddings_np, embedding_model)
                processed_output = process_triple(consistent_output)
            
            # 写入最终结果
            f.write(processed_output + '\n')
    
    print(f"\nPrediction finished. Results saved to {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()