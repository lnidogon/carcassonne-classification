# python dataset/partition.py
import os, random
root = "./dataset/augmented/"
traini = []
testi = []
evali = []
for file in os.listdir(root):
    option = random.choices([traini, testi, evali], weights=[70, 15, 15])
    option[0].append(file)
with open('./dataset/train_annotations.txt', 'w') as f:
    for file in traini:
        f.write(os.path.basename(file) + '\n')
with open('./dataset/test_annotations.txt', 'w') as f:
    for file in testi:
        f.write(os.path.basename(file) + '\n')
with open('./dataset/eval_annotations.txt', 'w') as f:
    for file in evali:
        f.write(os.path.basename(file) + '\n')