#!/usr/bin/env python
# coding: utf-8
number_of_cluster = 57
batch_size = 30
how_many_from_a_batch = 30
modelSize = "base"


import os
missing_classes = [5, 7, 11, 15, 17, 19, 21, 23, 25, 29, 33, 35, 45, 47, 50, 51, 55, 57, 59, 61, 63, 65, 69, 70, 73, 75]


import csv
csv_file_path = "./dataset/augmented_annotations.csv"
image_class = {}
with open(csv_file_path, mode='r', newline='') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        image_class[row['image_name']] = row['type']


with open('./dataset/test_annotations.txt', 'r') as f:
    testi = [x.strip('\n') for x in f.readlines()]
with open('./dataset/train_annotations.txt', 'r') as f:
    traini = [x.strip('\n') for x in f.readlines()]
with open('./dataset/eval_annotations.txt', 'r') as f:
    evali = [x.strip('\n') for x in f.readlines()] 


# In[10]:


import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
image_filenames = []

def load_images_from_folder(folder, whichFiles = []):
    images = []
    image_filenames = []
    for filename in os.listdir(folder):
        if filename not in whichFiles:
            continue
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB') 
            if(int(filename[5:11]) % (batch_size/how_many_from_a_batch) == 0):
                images.append(img)
                image_filenames.append(filename)
    return images, image_filenames

image_folder = './dataset/augmented/'
train_images, image_filenames = load_images_from_folder(image_folder, traini)
test_images, _ = load_images_from_folder(image_folder, testi)

image_processor = AutoImageProcessor.from_pretrained(f"facebook/dinov2-{modelSize}")
model = Dinov2Model.from_pretrained(f"facebook/dinov2-{modelSize}")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded")

def preprocess_image(image):
    image = image.resize((256, 256))
    img_arr = np.asarray(image)
    img_arr = img_arr.astype('float32') / 1.0
    return img_arr

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = preprocess_image(img)

        processed_images.append(img)
    return processed_images

#print(len(train_images))
processed_images = preprocess_images(train_images)

embeddings = []
with torch.no_grad():
    for i, image in enumerate(processed_images):
        #print(i)
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        embedding = model(**inputs).last_hidden_state[:, 0, :]
        embeddings.append(embedding.detach().cpu())
embeddings = torch.cat(embeddings, dim=0)


embeddings_np = embeddings.numpy()

print("Calculated embeddings")

kmeans = KMeans(n_clusters=number_of_cluster, random_state=23072003) 
clusters = kmeans.fit_predict(embeddings_np)

print("Predicted clusters using k-means")


# In[ ]:


clusters = kmeans.fit_predict(embeddings_np)

image_cluster_mapping = {filename : cluster for filename, cluster in zip(image_filenames, clusters)}
#print(len(list(image_cluster_mapping.keys())))
#print(len(image_filenames))

all_images_in_a_cluster = {}

all_clusters=set()

class_frequencies = {}
total_images = 0

for fn in image_filenames:
    if image_class[fn] not in class_frequencies:
        class_frequencies[image_class[fn]] = 0
    class_frequencies[image_class[fn]]+=1
    total_images += 1

    #print(image_cluster_mapping[fn])
    all_clusters.add(image_cluster_mapping[fn])
    #print(f"Image: {fn} is in Cluster: {image_cluster_mapping[fn]}")
    if image_cluster_mapping[fn] not in all_images_in_a_cluster:
        all_images_in_a_cluster[image_cluster_mapping[fn]] = []
    all_images_in_a_cluster[image_cluster_mapping[fn]].append(fn)

print("Calculated class frequencies for all classes")

# In[13]:


from numpy import sort
#print(total_images)
sorted_indexes = sorted(class_frequencies, key=class_frequencies.get)
#for i in sorted_indexes:
#    print(f"{i} {class_frequencies[i]}")


# In[ ]:


def class_to_description(c):
    ans = []
    if c == 81:
        return ['T', 'G', 'T', 'G']
    elif c == 82:
        return ['G', 'T', 'G', 'T']
    for i in range(4):
        if c%3 == 0: 
            ans.append('G')
        if c%3 == 1:
            ans.append('R')
        if c%3 == 2:
            ans.append('T')
        c //= 3
    return ans[::-1]


# In[15]:


from collections import Counter
from numpy import sort
all_mods = []
highest_chance = {}
highest_chance_index = {}
class_cluster_freq = []
class_cluster_occurances = []

for i in range(number_of_cluster):
    print(f"Lets {i}-ing go")
    all_classes_in_a_cluster = []
    for image in all_images_in_a_cluster[i]:
        all_classes_in_a_cluster.append(image_class[image])
        #print(f"{image}: {image_class[image]}",end=' ')
    temp = Counter(all_classes_in_a_cluster)
    temp2 = Counter(all_classes_in_a_cluster)
    print()
    print(temp)
    class_cluster_occurances.append(temp2)
    for j in range(83):
        if str(j) in temp:
            temp[str(j)] /= class_frequencies[str(j)]
            if str(j) not in highest_chance:
                highest_chance[str(j)] = 0
            if temp[str(j)] > highest_chance[str(j)]: 
                highest_chance[str(j)] = temp[str(j)]
    print(temp)
    class_cluster_freq.append(temp)
    print(max(temp, key=temp.get))
    all_mods.append(max(temp, key=temp.get))
all_mods = sorted(all_mods, key=float)
print(all_mods)



# In[16]:

'''
for i in range(83):
    if str(i) not in all_mods and i not in missing_classes:
        print(f"{i} _ {class_to_description(i)}")
for i in range(83):
    if str(i) in highest_chance:
        print(f"{str(i)} - {highest_chance[str(i)]}")
'''


# In[17]:


def classify_image(image_path, image_processor, model, kmeans, device):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))  
    with torch.no_grad():
        inputs = image_processor(images=img, return_tensors="pt").to(device)
        embedding = model(**inputs).last_hidden_state[:, 0, :]
        embedding = embedding.detach().cpu().numpy()




    cluster = kmeans.predict(embedding)[0]

    return cluster

'''
custom_image_path = "../Cutouts/primjer_igre_11/7_1.jpg"
predicted_cluster = classify_image(
    custom_image_path, 
    image_processor, 
    model,  
    kmeans, 
    device
)

print(f"The image belongs to cluster: {predicted_cluster}")
for image in all_images_in_a_cluster[predicted_cluster]:
    print(f"{image}: {image_class[image]} / {' '.join(class_to_description(int(image_class[image])))}")
'''

# In[18]:


import math
ith_class = []
cost_matrix = []
for i in range(83):
    if i in missing_classes:
        continue
    ith_class.append(i)
    temp = [1000000] * number_of_cluster
    for j in range(number_of_cluster):
        if str(i) not in class_cluster_freq[j]:
            continue
        temp[j] = (1 - class_cluster_freq[j][str(i)])**4
    cost_matrix.append(temp)

print("Calculated the cost matrix")
print(cost_matrix)


# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html

from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost_matrix)

print("Solved the linear sum assignment problem")

# In[21]:


cluster2class : dict = {}
for i in range(len(ith_class)):
    print(f"{ith_class[col_ind[i]]}: {ith_class[i]} - {class_cluster_freq[col_ind[i]][str(ith_class[i])]}")
    cluster2class[col_ind[i]] = ith_class[i]

print("Saved thee cluster to class mapping")

torch.save(cluster2class, f"./models/cluster2class_{modelSize}.pt")


correct = 0
total = 0
for i in range(57):
    #print("I", i)
    index = ith_class[i]
    #print("Index", index)
    #print(class_cluster_occurances[col_ind[i]])
    #print("Col ind", col_ind[i])
    for j in class_cluster_occurances[col_ind[i]]:
        if j == str(index):
            correct += class_cluster_occurances[col_ind[i]][j]
        total += class_cluster_occurances[col_ind[i]][j]
    #print(correct, total)
print("Accuracy on the test set", f"{correct}/{total} = {correct/total}")


import joblib
joblib.dump(kmeans, f"./models/kmeans_model_{modelSize}.pkl")

print("DONE")
