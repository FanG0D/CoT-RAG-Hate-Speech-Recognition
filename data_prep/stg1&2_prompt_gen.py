import json
import re

def create_stage1_sft_data(raw_data_path, output_path):
    """
    Args:
        raw_data_path (str): 原始数据文件的路径 (e.g., 'train.json')。
        output_path (str): 生成的SFT训练文件的路径 (e.g., 'stage1_sft_train.jsonl')。
    """


    prompt_template = (
        "你是一个会思考的语言审核专家，请你分析我的句子并且从中提取出一个或者多个（不超过三个）三元组，"
        "其中三元组的最后一个元素严格来自于[Racism, Region, LGBTQ, Sexism, others, non-hate]，"
        "请注意可能出现的谐音、生僻词以及敏感词。三元组之间用[SEP]分隔，以[END]结尾，"
        "只需要返回三元组的内容，且不需要任何前缀\n"
        "### 句子：\n{sentence}\n"
        "### 三元组：\n"
    )

    processed_data = []
    
    # 读取原始JSON文件
    try:
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {raw_data_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {raw_data_path} 不是有效的JSON格式。")
        return

    # 遍历每一条原始数据
    for item in raw_data:
        # 提取需要的内容
        sentence = item.get("content")
        target_output = item.get("output")

        if sentence is None or target_output is None:
            print(f"警告：跳过一条不完整的记录: {item}")
            continue

        # 使用模板构建完整的instruction + input部分
        full_prompt = prompt_template.format(sentence=sentence)
        
        sft_record = {
            "instruction": "",  # 留空，因为您的示例中它是空的
            "input": full_prompt, # 将完整的Prompt放入input字段
            "output": target_output
        }
        
        processed_data.append(sft_record)

    # 将处理好的数据写入新的JSONL文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in processed_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"成功处理 {len(processed_data)} 条数据，并保存到 {output_path}")




def create_stage2_cot_data(raw_data_path, output_path):
    """
    将DeepSeek生成的原始CoT回答转换为Stage 2 CoT SFT所需的JSONL格式。

    Args:
        raw_data_path (str): DeepSeek生成的原始数据文件的路径 (e.g., 'deepseek_generated.jsonl')。
        output_path (str): 生成的CoT SFT训练文件的路径 (e.g., 'stage2_cot_train.jsonl')。
    """
    
    # 与Stage 1保持一致的Prompt模板，用于构建最终的input字段
    prompt_template = (
        "你是一个会思考的语言审核专家，请你分析我的句子并且从中提取出一个或者多个（不超过三个）三元组，"
        "其中三元组的最后一个元素严格来自于[Racism, Region, LGBTQ, Sexism, others, non-hate]，"
        "请注意可能出现的谐音、生僻词以及敏感词。三元组之间用[SEP]分隔，以[END]结尾，"
        "只需要返回三元组的内容，且不需要任何前缀，注意思维格式\n"
        "### 句子：\n{sentence}\n"
        "### 三元组：\n"
    )

    # 正则表达式，用于从DeepSeek的输出中提取思维和结果
    # re.DOTALL 使得 '.' 可以匹配包括换行符在内的任意字符
    cot_pattern = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)

    processed_data = []

    try:
        with open(raw_data_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"警告: 跳过一行无效的JSON: {line.strip()}")
                    continue

                sentence = item.get("content")
                deepseek_response = item.get("deepseek_response")

                if not sentence or not deepseek_response:
                    print(f"警告: 跳过一条不完整的记录: {item}")
                    continue

                # 使用正则表达式解析DeepSeek的输出
                match = cot_pattern.search(deepseek_response)
                
                if not match:
                    print(f"警告: 无法从以下响应中解析CoT结构，跳过记录: {deepseek_response}")
                    continue
                
                # 提取思维过程和最终结果
                # group(1) 是 <think> 标签内的内容
                # group(2) 是 </think> 标签后的内容
                thinking_process = match.group(1).strip()
                final_result = match.group(2).strip()

                # 构建用于CoT微调的输出，保留<think>标签结构，让模型学会生成这种完整的带标签的格式
                cot_output = f"<think>\n{thinking_process}\n</think>\n{final_result}"

                # 构建完整的SFT记录
                full_prompt_input = prompt_template.format(sentence=sentence)
                
                sft_record = {
                    "instruction": "",
                    "input": full_prompt_input,
                    "output": cot_output  # 目标输出现在包含了完整的CoT结构
                }
                
                processed_data.append(sft_record)

    except FileNotFoundError:
        print(f"错误：找不到文件 {raw_data_path}")
        return

    # 将处理好的数据写入新的JSONL文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for record in processed_data:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"成功处理 {len(processed_data)} 条CoT数据，并保存到 {output_path}")

def