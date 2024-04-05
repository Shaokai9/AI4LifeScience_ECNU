#本代码是由于需要在论文附图中展示三维计数效果，故将其原先生成的txt文件按照来源分开

import os
import re


def split_txt_by_source(input_txt_path, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 创建一个字典来存储每个来源的内容
    source_dict = {}

    with open(input_txt_path, 'r') as f:
        for line in f:
            # 格式例如: "ID: 1, BBox (normalized): [x1, y1, w1, h1], Source: image1.png"
            bbox_match = re.search(r"BBox \(normalized\): \[(.*?),(.*?),(.*?),(.*?)\]", line)
            source_match = re.search(r"Source: (.+)$", line)

            if bbox_match and source_match:
                bbox_parts = bbox_match.groups()
                source = source_match.group(1).strip()
                line_content = f"0 {bbox_parts[0]} {bbox_parts[1]} {bbox_parts[2]} {bbox_parts[3]}\n"

                # 将行按来源分类
                if source not in source_dict:
                    source_dict[source] = []
                source_dict[source].append(line_content)

    # 遍历字典并写入文件
    for source, lines in source_dict.items():
        output_path = os.path.join(output_directory, source.replace('.png', '.txt'))
        with open(output_path, 'w') as f:
            f.writelines(lines)


# 使用函数
input_txt_path = r'C:\Users\SuQun\Desktop\ss\3'  # 修改为你的输入txt文件路径
output_directory = r'C:\Users\SuQun\Desktop\ss\output3'  # 修改为你希望输出txt文件的目录

split_txt_by_source(input_txt_path, output_directory)

