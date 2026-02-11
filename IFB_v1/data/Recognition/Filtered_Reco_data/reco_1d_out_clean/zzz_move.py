import os
import shutil

def copy_filtered_files(input_directory, output_directory, keywords):
    # 创建 filtered 目录，如果不存在的话
    os.makedirs(output_directory, exist_ok=True)
    
    # 遍历输入目录中的文件
    for filename in os.listdir(input_directory):
        if filename.endswith('.jsonl') and any(keyword in filename for keyword in keywords):
            # 文件名包含指定的关键词，复制文件到 filtered 目录
            source_file = os.path.join(input_directory, filename)
            destination_file = os.path.join(output_directory, filename)
            shutil.copy(source_file, destination_file)
            print(f"Copied: {filename}")

if __name__ == "__main__":
    # 设置输入目录和输出目录
    input_directory = "./"  # 修改为你的文件目录路径
    output_directory = "../filtered/"
    
    # 指定需要检查的关键词
    keywords = ['female', 'male', 'black', 'white', 'young', 'adult', 'old']
    
    # 执行文件过滤和复制操作
    copy_filtered_files(input_directory, output_directory, keywords)

