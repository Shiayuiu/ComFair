import json

def clean_data(file_a, file_b, output_file):
    # 打开A和B文件
    with open(file_a, 'r', encoding='utf-8') as f_a, open(file_b, 'r', encoding='utf-8') as f_b:
        # 读取A文件和B文件的所有行
        data_a = [json.loads(line) for line in f_a]
        data_b = [json.loads(line) for line in f_b]

    # 确保B文件有足够的行数
    if len(data_b) < len(data_a):
        raise ValueError("B文件的行数少于A文件的行数，无法按要求进行清洗。")

    # 处理每一行数据
    for i, item_a in enumerate(data_a):
        pair_id = item_a.get("pair_id")
        pair_id = int(pair_id)
        if pair_id is not None:
            # 找到pair_id+1在B文件中的行
            corresponding_b_item = data_b[pair_id] if pair_id + 1 < len(data_b) else None
            if corresponding_b_item:
                # 获取B文件中对应行的context_condition字段
                context_condition = corresponding_b_item.get("context_condition")
                if context_condition:
                    # 将context_condition添加到A文件的对应行
                    item_a["context_condition"] = context_condition

    # 将清洗后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data_a:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

# 使用示例
clean_data('race_age.jsonl', 'Race_ethnicity.jsonl', 'race_age_sel.jsonl')
