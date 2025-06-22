import sys, os
from .Classificator import Classificator
from .ResnetMetricClassificator import ResnetMetricClassificator
from .CustomMetricClassificator import CustomMetricClassificator
from .KMeansClassificator import KMeansClassificator
from .DinoMetricClassificator import DinoMetricClassificator
import torch, csv, numpy as np
from PIL import Image

def class_to_description(c):
    ans = []
    for i in range(4):
        if c%3 == 0: 
            ans.append('G')
        if c%3 == 1:
            ans.append('R')
        if c%3 == 2:
            ans.append('T')
        c //= 3
    return ans[::-1]

def reconstructBoard(cutoutsPath : str, protoPath : str, classificator : Classificator, device, tileW : int = 256, tileH : int = 256):
    board : list[list] = []
    fileList = os.listdir(cutoutsPath)
    print(fileList)
    fileList.sort(key  = lambda x : tuple([int(y) for y in x.removesuffix('.jpg').split('_')]))
    for cutoutName in fileList:
        print(cutoutName)
        x, y = [int(x) for x in cutoutName.removesuffix('.jpg').split('_', 1)]
        while len(board) <= x:
            board.append([])
        while len(board[x]) <= y:
            board[x].append(None)
        cls = classificator.classify(os.path.join(cutoutsPath, cutoutName), device)
        board[x][y] = cls
    boardW = len(max(board, key = lambda x : len(x)))
    boardH = len(board)
    finalImage = Image.new('RGB', (tileW * boardW, tileH * boardH))
    for i in range(boardH):
        for j in range(len(board[i])):
            cls = board[i][j]
            if cls == None: 
                continue
            print(f"{i} {j} -> {cls} / {''.join(class_to_description(cls))}")

            desc = ''.join(class_to_description(cls))
            for k in range(4):
                if cls < 81:
                    protoName = desc + "01.png"
                else:
                    protoName = "TGTG02_.png"
                if os.path.isfile(os.path.join(protoPath, protoName)):
                    tile = Image.open(os.path.join(protoPath, protoName)).convert('RGB')
                    tile = tile.resize((256, 256))
                    if cls == 82:
                        k = 1
                    for _ in range(k):
                        tile = tile.transpose(Image.Transpose.ROTATE_90)
                    finalImage.paste(tile, (tileW * j, tileH * i))   
                    break
                desc = desc[1:] + desc[0]
    savePath = os.path.join(cutoutsPath, "..", "..", "output")
    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    finalImage.save(os.path.join(savePath, f"board_{sys.argv[2].split('.')[0]}_{sys.argv[1]}.png"))

#python -m Classify.measurePerformance custom-metric


opt = sys.argv[1]

current_dir = os.path.dirname(os.path.abspath(__file__))

imagesPath = os.path.join(current_dir,"../../dataset/augmented/")

testImagesPath = os.path.join(current_dir,"../../dataset/test_annotations.txt")

resnetPretrainedModelLocation = os.path.join(current_dir,"../../models/model_ResNet50-ImageNet-1k.pt")
resnetPretrainedRepsLocation = os.path.join(current_dir,"../../models/reps_ResNet50-ImageNet-1k.pt")

resnetModelLocation = os.path.join(current_dir,"../../models/model_ResNet50.pt")
resnetRepsLocation = os.path.join(current_dir,"../../models/reps_ResNet50.pt")

dinoSmallModelLocation = os.path.join(current_dir,"../../models/model_DINOv2-small.pt")
dinoSmallRepsLocation = os.path.join(current_dir,"../../models/reps_DINOv2-small.pt")

dinoBaseModelLocation = os.path.join(current_dir,"../../models/model_DINOv2-base.pt")
dinoBaseRepsLocation = os.path.join(current_dir,"../../models/reps_DINOv2-base.pt")


customModelLocation = os.path.join(current_dir,"../../models/model_simple.pt")
customRepsLocation = os.path.join(current_dir,"../../models/reps_simple.pt")

kmeansBaseLocation = os.path.join(current_dir,"../../models/kmeans_model_base.pkl")
cluster2classBaseLocation = os.path.join(current_dir,"../../models/cluster2class_base.pt")

annotationsLocation = os.path.join(current_dir,"../../dataset/augmented_annotations.csv")


numberOfClasses = 83

image_class = {}
with open(annotationsLocation, mode='r', newline='') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        image_class[row['image_name']] = row['type']
with open(testImagesPath, 'r') as f:
    testImageNames = [x.strip('\n') for x in f.readlines()]

model : Classificator
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelDict : dict[Classificator] = {
    "ResNet50-ImageNet-1k" : ResnetMetricClassificator(
        modelLoaction=resnetPretrainedModelLocation, 
        repsLocation=resnetPretrainedRepsLocation, 
        pretrained=True,
        device=device),
    "ResNet50" : ResnetMetricClassificator(
        modelLoaction=resnetModelLocation, 
        repsLocation=resnetRepsLocation, 
        pretrained=False,
        device=device),
    "simple" : CustomMetricClassificator(
        modelLoaction=customModelLocation,
        repsLocation=customRepsLocation,
        device=device),
    "DINOv2-small" : DinoMetricClassificator(
        modelLoaction=dinoSmallModelLocation,
        repsLocation=dinoSmallRepsLocation,
        size="small",
        device=device),
    "DINOv2-base" : DinoMetricClassificator(
        modelLoaction=dinoBaseModelLocation,
        repsLocation=dinoBaseRepsLocation,
        size="base",
        device=device),
    "kmeans-base" : KMeansClassificator(
        kmeans_location=kmeansBaseLocation,
        cluster2class_location=cluster2classBaseLocation,
        device=device
    )
}
classificator : Classificator = modelDict[opt]


if sys.argv[2] == "measure":
    confusionMatrix = np.zeros((numberOfClasses, numberOfClasses))
    for image in os.listdir(imagesPath):
        if image not in testImageNames:
            continue
        target = int(image_class[image])
        output = int(classificator.classify(os.path.join("dataset", "augmented", image), device))
        confusionMatrix[output][target] += 1

    numberOfImages = sum(sum(confusionMatrix[:,:]))

    TPsum = 0

    precision = []
    recall = []
    accuracy = 0

    for i in range(numberOfClasses):
        TP = confusionMatrix[i][i]
        FP = sum(confusionMatrix[:,i])
        FN = sum(confusionMatrix[i,:])
        TN = numberOfImages - TP - FP - FN
        precision.append(TP / (TP + FP))
        recall.append(TP / (TP + FN))
        accuracy += TP
    accuracy /= numberOfImages


    print(f"[ACCURACY]: {accuracy}")
    print("[PRECISION]: ", precision)
    print("[RECALL]: ", recall)
    print("[AVERAGE PRECISION]: ", np.nanmean(precision))
    print("[AVERAGE RECALL]: ", np.nanmean(recall))
elif sys.argv[2] != None:
    cutoutsLocation = os.path.join(current_dir,f"../../cutouts/{sys.argv[2].split('.')[0]}")
    protoLocation = os.path.join(current_dir,"../../dataset/scanned_tiles/")
    reconstructBoard(cutoutsLocation, protoLocation, classificator, device)

