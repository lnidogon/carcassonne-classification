import sys
import time
import torch.optim
from torch.utils.data import DataLoader
from CustomModel import SimpleMetricEmbedding
from utils import train, evaluate, compute_representations
import os
from dataset import CarcMetricDataset
from ResnetModel import ResNet50Model
from DinoModel import DinoModel
import numpy as np

EVAL_ON_TEST = True
EVAL_ON_TRAIN = True


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    dataset_root = "./dataset/augmented/"
    traini = []
    testi = []
    trainevali = []
    with open('./dataset/test_annotations.txt', 'r') as f:
        testi = [x.strip('\n') for x in f.readlines()]
    with open('./dataset/train_annotations.txt', 'r') as f:
        traini = [x.strip('\n') for x in f.readlines()]
    with open('./dataset/eval_annotations.txt', 'r') as f:
        evali = [x.strip('\n') for x in f.readlines()]    
    print("done reading")
    ds_train = CarcMetricDataset(dataset_root, split='train', whichFiles=traini)
    ds_test = CarcMetricDataset(dataset_root, split='test', whichFiles=testi)
    ds_traineval = CarcMetricDataset(dataset_root, split='traineval', whichFiles=traini + evali)
    
    
    num_classes = 83

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size= 32 if sys.argv[1] != "dino" else 16,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    lrn = 1e-3
    epochs = 11
    if sys.argv[1] == 'ResNet50-ImageNet-1k':
        emb_size = 64
        model = ResNet50Model(emb_size, pretrained=True).to(device) 
        lrn=1e-5
    elif sys.argv[1] == 'ResNet50':
        emb_size = 64
        lrn = 1e-5
        model = ResNet50Model(emb_size, pretrained=False).to(device)
    elif sys.argv[1] == 'simple':
        emb_size = 256
        model = SimpleMetricEmbedding(3, emb_size).to(device)
        lrn = 1e-4
    elif sys.argv[1] == "DINOv2-small":
        emb_size = 64
        lrn = 1e-5
        model = DinoModel(emb_size, "small").to(device)
    elif sys.argv[1] == "DINOv2-base":
        emb_size = 64
        model = DinoModel(emb_size, "base").to(device)
        lrn = 1e-5
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lrn
    )

    all_train_losses = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.time_ns()
        train_loss, train_losses = train(model, optimizer, train_loader, device)
        print(train_loss)
        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
        if EVAL_ON_TRAIN or EVAL_ON_TEST:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)

        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")

        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            acc1 = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
        all_train_losses.append(train_losses)
        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")

    np.save(f"./visualise/loss_{sys.argv[1]}.npy", all_train_losses)
    torch.save(model.state_dict(), f"./models/model_{sys.argv[1]}.pt")
    representations = compute_representations(model, train_loader, num_classes, emb_size, device)
    torch.save(representations, f"./models/reps_{sys.argv[1]}.pt")

