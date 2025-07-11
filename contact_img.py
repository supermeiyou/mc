from PIL import Image
import os


def create_collage(image_paths, output_path, cols=2):
    """创建照片拼贴"""
    # 打开所有图像
    images = [Image.open(img_path) for img_path in image_paths]

    # 确保所有图像尺寸一致（1200×600）
    for i, img in enumerate(images):
        if img.size != (1200, 600):
            print(f"警告：图像 {i + 1} 尺寸不是1200×600，正在调整...")
            images[i] = img.resize((1200, 600))

    # 分为两列（前5个一列，后5个一列）
    left_column = images[:5]
    right_column = images[5:]

    # 计算拼贴尺寸
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # 创建拼贴画布（两列，每列5张）
    collage_width = max_width * 2
    collage_height = max_height * 5
    collage = Image.new('RGB', (collage_width, collage_height))

    # 拼接左列
    for i, img in enumerate(left_column):
        position = (0, i * max_height)
        collage.paste(img, position)

    # 拼接右列
    for i, img in enumerate(right_column):
        position = (max_width, i * max_height)
        collage.paste(img, position)

    # 缩放至合适大小
    target_width = 2560  # 目标宽度，可根据需要调整
    scale_factor = target_width / collage_width
    if scale_factor < 1:
        new_size = (int(collage_width * scale_factor), int(collage_height * scale_factor))
        collage = collage.resize(new_size, Image.LANCZOS)

    # 保存结果
    collage.save(output_path)
    print(f"拼贴已保存至: {output_path}")


if __name__ == "__main__":
    # 设置图片路径（假设图片在当前目录的images文件夹中）
    image_folder = "image"
    image_files = [f"LSTM-90-{i}.png" for i in range(1, 6)]  # 图片命名为img1.jpg, img2.jpg, ..., img10.jpg
    for i in range(1, 6):
        image_files.append(f"LSTM-365-{i}.png")
    image_paths = [os.path.join(image_folder, fname) for fname in image_files]

    # 检查所有图片是否存在
    missing_images = [path for path in image_paths if not os.path.exists(path)]
    if missing_images:
        print("错误：以下图片不存在:")
        for path in missing_images:
            print(f"  - {path}")
        exit(1)

    # 创建拼贴
    output_path = "collage.jpg"
    create_collage(image_paths, output_path)