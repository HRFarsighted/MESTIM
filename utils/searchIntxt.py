import os

def find_files_with_content(folder_path, target_content):
    matching_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if target_content in line:
                        matching_files.append(file)
                        break
    return matching_files

folder_path = '/home/hour/hr/Hour/TimeMachine-main/TimeMachine_supervised/logs/LongForecasting/ETTh1/8.26'  # 替换为实际的文件夹路径
target_content = 'mse:0.38'  # 替换为你要查找的特定内容
result = find_files_with_content(folder_path, target_content)
print(result)