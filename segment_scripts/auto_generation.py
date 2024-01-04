from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

model_type = "vit_h"
sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
device = "cuda"

folder_path = "D:\\nerf_data\\360garden_4\\input"
save_folder_path = "D:\\nerf_data\\360garden_4\\mask\\"

def show_anns(anns,file_name):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3),dtype=np.uint8)
    #img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.randint(0,256,3)])
        img[m] = color_mask
    #ax.imshow(img)
    #plt.imshow(img)
    write_file_name = os.path.join(save_folder_path,file_name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imwrite(write_file_name, img)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam
)

files = os.listdir(folder_path)
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        image = cv2.imread(file_path)
        masks = mask_generator.generate(image)
        #plt.figure(figsize=(20,20))
        #plt.imshow(image)
        show_anns(masks,file_name)
        #plt.axis('off')
        #plt.show()