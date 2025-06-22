'''
    classes that don't exist:
     5 (GGRT),
     7 (GGTR),
    11 (GRGT),
    15 (GRTG),
    17 (GRTT),
    19 (GTGR),
    21 (GTRG),
    23 (GTRT),
    25 (GTTR),
    29 (RGGT),
    33 (RGTG),
    35 (RGTT),
    45 (RTGG),
    47 (RTGT),
    50 (RTRT),
    51 (RTTG),
    55 (TGGR),
    57 (TGRG),
    59 (TGRT),
    61 (TGTR),
    63 (TRGG),
    65 (TRGT),
    69 (TRTG),
    70 (TRTR),
    73 (TTGR),
    75 (TTRG),

    Rotacije: 
            4 x GGRT
            4 x GGTR
            4 x GRGT
            4 x GRTT
            4 x GTRT
            4 x GTTR
            2 x RTRT
            sum = 26

    Ukupno klasa = 83 - 26 = 57


'''

import os
import numpy as np
from PIL import Image

import csv
def load_images_from_folder(folder):
    images = []
    image_names = []
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')  
            images.append(img)
            image_names.append(filename)
    return images, image_names

image_folder = './dataset/scanned_tiles/'
images, image_names = load_images_from_folder(image_folder)


def resize_images(images):
    processed_images = []
    for img in images:
        img = img.resize((256, 256))  
        processed_images.append(img)
    return processed_images

images = resize_images(images)

import cv2
import numpy as np
import albumentations as A
from PIL import Image
import random

def random_resize(image, **kwargs):
    height = random.randint(40, 100)  
    width = random.randint(40, 100)   
    return cv2.resize(image, (width, height))

def hash_name(name):
    if len(name) == 5:
        return 81 + (1 if name[0] == 'G' else 0)
    val_dict = {'G': 0, 'R': 1, 'T': 2}
    ret = 0
    for side in name:
        ret *= 3
        ret += val_dict[side]
    return ret
        
    


def augment_image(image):
    transform = A.Compose([
        A.PadIfNeeded(min_height=500, min_width=500, position='center', border_mode=cv2.BORDER_REFLECT),
        A.Rotate(limit=(-5,5), p=1),                   
        A.Crop(x_min=100, y_min=100, x_max=400, y_max=400),  
        A.RandomCrop(width=256, height=256, p=1.0),  
        A.Lambda(image=random_resize, p=1.0),
        A.Resize(256, 256),
        A.RandomBrightnessContrast(),        
        A.Blur(p=1),       
    ])

    augmented = transform(image=image)
    return augmented['image']

number_of_instances = 60
save_path = "./dataset/augmented/"
image_index=0
with open(save_path + "../augmented_annotations.csv", "w", newline='') as aa_csv:
    field_names = ["image_name", "type"]
    writer = csv.DictWriter(aa_csv, fieldnames=field_names)
    writer.writeheader()
    for i, image in enumerate(images):
        image = np.array(image) 
        image_name = image_names[i][:-4]
        special = False
        if len(image_name) == 7:
            special = True
        image_name = image_name[:4]
        for j in range(4): 
            for i in range(number_of_instances):
                augmented_image = augment_image(image)
                file_name=f"TILE_{str(image_index).zfill(6)}.jpg"
                print(file_name)
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path + file_name, augmented_image)
                writer.writerow({'image_name': file_name,'type': hash_name(image_name + ('_' if special else ''))})
                image_index += 1
            image= cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image_name = image_name[1:] + image_name[:1]
processed_images = resize_images(images)
