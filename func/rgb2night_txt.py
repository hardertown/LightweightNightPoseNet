import cv2
import os

def nightize(img):
    # 对灰度图像进行高斯模糊
    blurred_image = cv2.GaussianBlur(img, (15, 15), 0)
    # 调整对比度和亮度
    adjusted_image = cv2.convertScaleAbs(blurred_image, alpha=1.0, beta=-20)
    return adjusted_image


# 输入文件和输出文件夹路径
# input_txt = r'E:\mypose\down_train.txt'
# output_folder = r'E:\mycoco\down_data\merged_data1'
input_txt = r'E:\mypose\down_valid.txt'
output_folder = r'E:\mycoco\down_data\merged_data_valid1'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 打开输入文件
with open(input_txt, 'r') as file:
    # 逐行读取文件内容
    for i, line in enumerate(file):
        # 分割每行数据
        data = line.strip().split(' ')

        # 图像路径
        image_path = data[0]
        # 关键点坐标
        points = data[1:]

        # 读取图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 进行夜视处理
        night_vision_img = nightize(img)

        # 构建输出文件路径
        output_file = os.path.join(output_folder, f'{str(i + 1).zfill(3)}.jpg')

        # 保存处理后的图像
        cv2.imwrite(output_file, night_vision_img)

        print("Night vision image saved to:", output_file)

