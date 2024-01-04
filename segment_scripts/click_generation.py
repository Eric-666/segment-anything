import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
import os

'''
步骤1: 查看测试图片显示前景和背景的标记点
      鼠标左键为选前景点，鼠标右键为选背景点，关闭图像退出选点
'''
# 定义回调函数，用于处理鼠标点击事件
def onclick_points(event):
    if event.button == 1:  # 鼠标左键
        label = 1
    elif event.button == 3:  # 鼠标右键
        label = 0
    else:
        return
    x = event.xdata
    y = event.ydata
    # 将点击的点的坐标和类型保存到列表中
    points_coords.append([x, y])
    labels_coords.append(label)
    # 在图像上绘制点击的点
    plt.title("choose pos_points and neg_points")
    plt.scatter(x, y, marker='.', s=200, c='red' if int(label) == 1 else 'blue')
    plt.draw()
    # 打印保存的点坐标和类型
    print(f"坐标：({x}, {y})，类型：{label}")



def show_pic_choose_points(img):
    global points_coords
    global labels_coords
    # 创建一个新的图像窗口并显示图像
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("choose pos_points and neg_points")
    # 定义存储点坐标和类型的列表
    points_coords = []
    labels_coords = []
    # 绑定鼠标点击事件处理函数
    cid = fig.canvas.mpl_connect('button_press_event', onclick_points)
    # 显示图像和等待用户点击点
    plt.show()
    return points_coords, labels_coords


def show_points(coords, labels, ax):
    # 筛选出前景目标标记点
    pos_points = coords[labels == 1]
    # 筛选出背景目标标记点
    neg_points = coords[labels == 0]
    # x-->pos_points[:, 0] y-->pos_points[:, 1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='.', s=200)  # 前景的标记点显示
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='blue', marker='.', s=200)  # 背景的标记点显示


def show_mask(mask, ax, random_color=False):
    if random_color:    # 掩膜颜色是否随机决定
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# 加载模型
sam_checkpoints_b = "./checkpoints/sam_vit_b_01ec64.pth"
sam_checkpoints_l = "./checkpoints/sam_vit_l_0b3195.pth"
sam_checkpoints_h = "./model/sam_vit_h_4b8939.pth"
# 模型类型
model_type_b = "vit_b"
model_type_l = "vit_l"
model_type_h = "vit_h"
device = "cuda"
# folder
folder_path = "D:\\nerf_data\\plant_and_book\\input\\"
save_folder_path = "D:\\nerf_data\\plant_and_book\\mask\\"
multimask_output_bool = False

if __name__ == "__main__":
    #初始化模型
    sam = sam_model_registry[model_type_h](sam_checkpoints_h) 
    sam.to(device=device)
    predictor = SamPredictor(sam)

    files = os.listdir(folder_path)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            predictor.set_image(image)

            # 选点
            points, labels = show_pic_choose_points(image)
            input_points, input_labels = np.array(points.copy()), np.array(labels.copy())  # input_points(n,2) input_labels(n,)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,     # 形式 [[x,y], [x,y],...] (n,2)
                point_labels=input_labels,      # 形式 [0, 1, ...] (n, )
                multimask_output=multimask_output_bool, 
            )

            write_file_name = os.path.join(save_folder_path,file_name)
            # =============显示预测
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))

                #show_mask(mask, plt.gca())
                #plt.savefig('C:\\Users\\xzx\\Desktop\\example.png')

                plt.imshow(image)
                show_mask(mask,plt.gca())
                show_points(input_points, input_labels, plt.gca())
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()

                #保存图片
                converted_array = np.zeros_like(mask, dtype=np.uint8)
                converted_array[mask] = 255
                converted_array[:,:,np.newaxis]
                gray_image = cv2.cvtColor(converted_array, cv2.COLOR_GRAY2BGR)                
                #cv2.imwrite(write_file_name, gray_image)
                print("done")