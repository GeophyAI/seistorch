import os
import re
import zipfile

def pack_files(directory, target_file):
    max_number = float('-inf')
    max_files = []
    yml_files = []

    # 遍历目录及其子文件夹下的文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # 匹配以 .yml 结尾的文件
            m1 = file.endswith('.yml')
            m2 = file.startswith('model_error')
            m3 = file == "loss.npy"
            if  any([m1, m2, m3]):
                yml_files.append(file_path)

            # 匹配文件名中的两个数字
            match = re.search(r'para(vs|vp|rho)F(\d+)E(\d+).npy', file)
            if match:
                number_1 = int(match.group(2))
                number_2 = int(match.group(3))

                # 比较数字的大小，更新最大值和对应的文件
                current_number = number_1 * 100 + number_2
                if current_number > max_number:
                    max_number = current_number
                    max_files = [file_path]
                elif current_number == max_number:
                    max_files.append(file_path)

    # 创建一个zip文件
    with zipfile.ZipFile(target_file, 'w') as zipf:
        # 将以 .yml 结尾的文件添加到zip文件中
        for file_path in yml_files:
            zipf.write(file_path, arcname=os.path.relpath(file_path, directory))

        # 将具有最大数字的文件添加到zip文件中
        for file_path in max_files:
            zipf.write(file_path, arcname=os.path.relpath(file_path, directory))

# 指定目录和输出zip文件名
directory = '/mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss'
output_zip = '/mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss/output.zip'

# 执行打包操作
pack_files(directory, output_zip)
