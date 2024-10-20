import os
import re
import sys

def modify_txt_files(directory):
    # 检查传入的路径是否有效
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在，请检查路径")
        return

    # 获取指定目录中的所有文件
    files = os.listdir(directory)

    # 正则匹配文件名，以Star开头，扩展名为.txt
    pattern = re.compile(r'^Star.*\.txt$')

    for file in files:
        # 构建文件的完整路径
        file_path = os.path.join(directory, file)

        # 如果文件名符合正则表达式，且是.txt文件
        if pattern.match(file) and os.path.isfile(file_path):
            try:
                # 读取文件内容
                with open(file_path, 'r') as f:
                    content = f.read()

                # 确保文件内容不为空
                if content:
                    # 修改第一个字符为 "6"
                    content = '5' + content[1:]

                    # 写回修改后的内容
                    with open(file_path, 'w') as f:
                        f.write(content)

                print(f"修改了文件: {file_path}")

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

if __name__ == '__main__':
    # 检查是否传递了目录路径参数

    if len(sys.argv) < 2:
        print("请提供要处理的目录路径，例如：python modify_files.py /path/to/directory")
    else:
        directory = sys.argv[1]
        modify_txt_files(directory)
