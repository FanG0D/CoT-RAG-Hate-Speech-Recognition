import json
from difflib import SequenceMatcher
from collections import defaultdict

def parse_line_to_quadruples(line: str) -> list[tuple]:
    """
    将模型输出的一行文本解析为四元组列表。
    例如："A | B | C | D [SEP] E | F | G | H [END]" -> [('A','B','C','D'), ('E','F','G','H')]
    """
    line = line.strip()
    if not line:
        return []
    
    # 去除结尾的 [END] 标记
    if line.endswith("[END]"):
        line = line[:-5].strip()
    
    quadruples = []
    parts = line.split("[SEP]")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        elements = [elem.strip() for elem in part.split('|')]
        # 确保每个部分都恰好是四个元素，不足或多余的都视为格式错误
        if len(elements) == 4:
            quadruples.append(tuple(elements))
        else:
            print(f"警告: 发现格式错误的元组，已跳过: '{part}'")
            
    return quadruples

def calculate_similarity(a: str, b: str) -> float:
    """
    计算两个字符串之间的相似度。
    """
    return SequenceMatcher(None, a, b).ratio()

def calculate_scores(predictions: list[list[tuple]], ground_truths: list[list[tuple]]) -> dict:
    """
    根据预测和标准答案，计算硬匹配和软匹配的分数。

    Args:
        predictions (list[list[tuple]]): 预测的四元组列表的列表，每个子列表对应一个样本。
        ground_truths (list[list[tuple]]): 标准答案的四元组列表的列表。

    Returns:
        dict: 包含硬软匹配P/R/F1分数的字典。
    """
    
    # 初始化各种计数器
    hard_true_positives = 0
    soft_true_positives = 0
    num_predictions = 0
    num_ground_truths = 0

    # 遍历每个样本
    for pred_quads, gt_quads in zip(predictions, ground_truths):
        num_predictions += len(pred_quads)
        num_ground_truths += len(gt_quads)
        
        # 为了避免重复匹配，使用一个集合来跟踪已经匹配过的标准答案
        gt_matched_hard = set()
        gt_matched_soft = set()

        # 对每个预测的四元组进行匹配检查
        for pred_quad in pred_quads:
            # --- 硬匹配检查 ---
            is_hard_matched = False
            for i, gt_quad in enumerate(gt_quads):
                if i not in gt_matched_hard and pred_quad == gt_quad:
                    hard_true_positives += 1
                    gt_matched_hard.add(i)
                    is_hard_matched = True
                    break # 找到一个匹配就停止，避免一个预测匹配多个标准答案
            
            # --- 软匹配检查 ---
            # 即使硬匹配成功，也要独立检查软匹配，因为一个预测可能硬匹配一个gt, 软匹配另一个gt
            is_soft_matched = False
            for i, gt_quad in enumerate(gt_quads):
                if i not in gt_matched_soft:
                    # 规则1: Targeted Group 和 Hateful 必须完全匹配
                    if pred_quad[2] == gt_quad[2] and pred_quad[3] == gt_quad[3]:
                        # 规则2: Target 和 Argument 的相似度均需超过0.5
                        sim_target = calculate_similarity(pred_quad[0], gt_quad[0])
                        sim_argument = calculate_similarity(pred_quad[1], gt_quad[1])
                        if sim_target > 0.5 and sim_argument > 0.5:
                            soft_true_positives += 1
                            gt_matched_soft.add(i)
                            is_soft_matched = True
                            break # 找到一个匹配就停止
    
    # --- 计算 P, R, F1 ---
    results = {}
    
    # 硬匹配分数
    hard_p = hard_true_positives / num_predictions if num_predictions > 0 else 0
    hard_r = hard_true_positives / num_ground_truths if num_ground_truths > 0 else 0
    hard_f1 = 2 * (hard_p * hard_r) / (hard_p + hard_r) if (hard_p + hard_r) > 0 else 0
    results['hard'] = {'P': hard_p, 'R': hard_r, 'F1': hard_f1}
    
    # 软匹配分数
    soft_p = soft_true_positives / num_predictions if num_predictions > 0 else 0
    soft_r = soft_true_positives / num_ground_truths if num_ground_truths > 0 else 0
    soft_f1 = 2 * (soft_p * soft_r) / (soft_p + soft_r) if (soft_p + soft_r) > 0 else 0
    results['soft'] = {'P': soft_p, 'R': soft_r, 'F1': soft_f1}

    # 平均F1分数
    avg_f1 = (hard_f1 + soft_f1) / 2
    results['average_f1'] = avg_f1
    
    return results

def main(prediction_file: str, ground_truth_file: str):
    """
    主函数，读取文件，执行评估并打印结果。
    """
    # 1. 读取并解析预测文件
    try:
        with open(prediction_file, 'r', encoding='utf-8') as f:
            pred_lines = f.readlines()
        predictions = [parse_line_to_quadruples(line) for line in pred_lines]
    except FileNotFoundError:
        print(f"错误: 预测文件未找到: {prediction_file}")
        return

    # 2. 读取并解析标准答案文件 (假设是JSON格式)
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        # 假设原始数据格式是 { "output": "A|B|C|D[SEP]...", ... }
        ground_truths = [parse_line_to_quadruples(item['output']) for item in gt_data]
    except FileNotFoundError:
        print(f"错误: 标准答案文件未找到: {ground_truth_file}")
        return
    except (json.JSONDecodeError, KeyError):
        print("错误: 标准答案文件格式不正确或缺少'output'字段。")
        return
        
    # 确保预测文件和答案文件行数一致
    if len(predictions) != len(ground_truths):
        print(f"警告: 预测文件行数 ({len(predictions)}) 与标准答案行数 ({len(ground_truths)}) 不匹配！")
        # 可以选择退出或继续评估匹配的部分
        # return 
    
    # 3. 计算分数
    scores = calculate_scores(predictions, ground_truths)
    
    # 4. 打印结果
    print("================ 评估结果 ================")
    print(f"硬匹配 (Hard Match):")
    print(f"  Precision: {scores['hard']['P']:.4f}")
    print(f"  Recall:    {scores['hard']['R']:.4f}")
    print(f"  F1-Score:  {scores['hard']['F1']:.4f}")
    print("-" * 40)
    print(f"软匹配 (Soft Match):")
    print(f"  Precision: {scores['soft']['P']:.4f}")
    print(f"  Recall:    {scores['soft']['R']:.4f}")
    print(f"  F1-Score:  {scores['soft']['F1']:.4f}")
    print("==========================================")
    print(f"最终平均 F1 分数: {scores['average_f1']:.4f}")
    print("==========================================")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="评估细粒度仇恨言论检测任务的F1分数")
    parser.add_argument("--pred_file", type=str, required=True, help="模型预测结果文件的路径 (每行一个输出)")
    parser.add_argument("--gt_file", type=str, required=True, help="标准答案文件的路径 (JSON格式)")
    
    args = parser.parse_args()
    
    main(args.pred_file, args.gt_file)