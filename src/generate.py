import cv2
import numpy as np
import glob
import os

img_dir='/media/khushal/679f068d-921b-4d14-890f-3081c1728f98/Rephrase/rephrase_data/test/*'
deg_dir='/media/khushal/679f068d-921b-4d14-890f-3081c1728f98/Rephrase/rephrase_data/degrade_test/'

  
def degrade(input_path: str, output_path: str) -> None:
        """Load image at `input_path`, distort and save as `output_path`"""
        SHIFT = 2
        image = cv2.imread(input_path)
        orig_img=image
        to_swap = np.random.choice([False, True], image.shape[:2], p=[.8, .2])
        swap_indices = np.where(to_swap[:-SHIFT] & ~to_swap[SHIFT:])
        swap_vals = image[swap_indices[0] + SHIFT, swap_indices[1]]
        image[swap_indices[0] + SHIFT, swap_indices[1]] = image[swap_indices]
        image[swap_indices] = swap_vals
        cv2.imwrite(output_path, image)


imgtypes= glob.glob(img_dir)
base_imgtypes=[os.path.basename(f) for f in imgtypes]
print(base_imgtypes)
for i in range(83):
        # os.makedirs(deg_dir+base_imgtypes[i])
        class_img=glob.glob(imgtypes[i]+"/*.jpg")
        img_name=[os.path.basename(f) for f in class_img]
        # print(img_name)
        for j in range(len(class_img)):
                f=class_img[j]
                degrade(f,deg_dir+base_imgtypes[i]+"/"+img_name[j])
        # print(len(class_img))

