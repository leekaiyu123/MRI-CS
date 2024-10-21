import os
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm  # 导入 tqdm 以显示进度条


def load_model(model_path, num_classes=2):
    # 加载 Faster R-CNN 模型
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model_weights = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in model_weights:
        model_weights = model_weights['model_state_dict']

    model.load_state_dict(model_weights, strict=False)  # 允许部分加载
    model.eval()
    return model


def get_transform():
    # 定义图像变换
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        # 如有必要，可添加归一化
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def save_cropped_image(image, box, save_folder, image_name):
    # 计算裁剪区域
    xmin, ymin, xmax, ymax = box
    cropped_image = image.crop((xmin, ymin, xmax, ymax))

    # 保存裁剪后的图像
    cropped_image_path = os.path.join(save_folder, f"{os.path.splitext(image_name)[0]}_crop.png")
    cropped_image.save(cropped_image_path)

def predict_and_display(model, image_folder, save_folder, threshold=0.5):
    transform = get_transform()

    # 确保保存文件夹存在
    os.makedirs(save_folder, exist_ok=True)

    # 获取所有图像文件
    image_files = os.listdir(image_folder)

    # 使用 tqdm 显示进度条
    for image_name in tqdm(image_files, desc="Processing", unit="image"):
        image_path = os.path.join(image_folder, image_name)
        try:
            image = Image.open(image_path).convert("RGB")  # 转换为 RGB
            image_transformed = transform(image).unsqueeze(0)  # 添加批处理维度

            with torch.no_grad():
                predictions = model(image_transformed)

            # 假设单图像批处理
            boxes = predictions[0]['boxes']
            scores = predictions[0]['scores']

            # 筛选低分数的预测
            keep = scores >= threshold

            if keep.sum() == 0:
                print(f"No detections above threshold for {image_name}")
                continue

            # 只选择得分最高的边界框
            max_score_index = scores[keep].argmax()
            best_box = boxes[keep][max_score_index].cpu().numpy()
            best_score = scores[keep][max_score_index].cpu().item()

            # 显示原始图像和边界框
            # fig, ax = plt.subplots(1)
            # ax.imshow(image)
            # plt.axis('off')  # 关闭坐标轴
            #
            # xmin, ymin, xmax, ymax = best_box
            # rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
            #                          linewidth=1, edgecolor='#00FF00', facecolor='none')
            # ax.add_patch(rect)
            # ax.text(xmin, ymin, f'{best_score:.2f}', color='white', fontsize=12,
            #         backgroundcolor='red')
            #
            # plt.title(f"Detection for {image_name}")
            # plt.show()

            # 保存裁剪后的图像
            save_cropped_image(image, best_box, save_folder, image_name)

        except Exception as e:
            print(f"Error processing {image_name}: {e}")


if __name__ == "__main__":
    model_path = 'faster_rcnn_model.pth'  # Path to the trained model
    image_folder = r'D:\control'  # Change this to your image folder path
    save_folder = r'D:\control_crop'  # Path to save cropped images

    model = load_model(model_path)
    predict_and_display(model, image_folder, save_folder, threshold=0.1)  # Adjust threshold as needed