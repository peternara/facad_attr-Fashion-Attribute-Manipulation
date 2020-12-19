import os
import numpy as np
import pickle
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm
from PIL import Image

import utils
import datasets
import models

num_epochs = 5000
SAVE_EMB = True
TRAIN_MODEL = True

data_dir = 'polo_sample'
model_dir = 'weights'
embeddings_dir = 'embeddings'
output_dir = 'output'

meta_dict = {}
os.makedirs(model_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])])
                                                                      
polo_color_dataset = datasets.PoloColorDataset(data_dir=data_dir,
                                        transform=data_transforms)
dataloader = DataLoader(polo_color_dataset,
                        shuffle=False,
                        num_workers=8,
                        batch_size=8)

colors_dict = polo_color_dataset.colors_dict


imageEncoder = models.ImageEncoder().to(device)
attrEncoder = models.AttrEmbeddingNetwork(len(colors_dict)).to(device)

criterion = nn.TripletMarginLoss(margin=0.2)
params = list(imageEncoder.parameters()) + list(attrEncoder.parameters())

optimizer = optim.Adam(params,lr = 0.0005)

def train_model(BASE_PATH):
    iteration_number = 0
    counter = []
    loss_history = []
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            img, pos ,neg, _ = data
            img, pos, neg = img.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()

            img_emb = imageEncoder(img)
            pos_emb = attrEncoder(pos)
            neg_emb = attrEncoder(neg)

            triplet_loss = criterion(img_emb,pos_emb,neg_emb)
            triplet_loss.backward()
            optimizer.step()

            if i %100 == 0 :
                iteration_number +=100
                counter.append(iteration_number)
                loss_history.append(triplet_loss.item())
                print("Epoch number {}\n Current loss {}\n".format(epoch,triplet_loss.item()))
        if epoch % 1000 == 0:
            torch.save(imageEncoder.state_dict(), os.path.join(BASE_PATH, f'img_encoder_{epoch}.pt'))
            torch.save(attrEncoder.state_dict(), os.path.join(BASE_PATH, f'attr_encoder_{epoch}.pt'))
    return imageEncoder, attrEncoder


def embed_all_images(dataloader, model, colors_dict, embeddings_dir):
    device = 'cpu'
    model.eval()
    model = model.to(device)
    emb_matrix = None
    img_list = []
    split = 0
    for i, data in tqdm.tqdm(enumerate(dataloader)):
        img, _ ,_, names = data
        img = img.to(device)
        img_emb = model(img)
        if emb_matrix is None:
            emb_matrix = img_emb
        else:
            emb_matrix = torch.cat((emb_matrix, img_emb),0)
        img_list += list(names)
        if emb_matrix.shape[0] >= 10:
            pickle.dump(emb_matrix, open(os.path.join(embeddings_dir, f'emb_matrix_{split}.pt'), 'wb'))
            pickle.dump(img_list, open(os.path.join(embeddings_dir, f'img_list_{split}.list'), 'wb'))
            split += 1
            emb_matrix = None
            img_list = []

    # embed the last split
    pickle.dump(emb_matrix, open(os.path.join(embeddings_dir, f'emb_matrix_{split}.pt'), 'wb'))
    pickle.dump(img_list, open(os.path.join(embeddings_dir, f'img_list_{split}.list'), 'wb'))
    return split

def evaluate_query_split(attr_emb, emb_matrix, k=10):
    euclidean_distance = F.pairwise_distance(attr_emb, emb_matrix, keepdim=True)
    return torch.topk(euclidean_distance, k=k,largest=False,dim=0)

def plot_topk(top_images, transforms, save_path=None):

    batch = None
    for img in top_images:
        img = Image.open(img)
        img = transforms(img).unsqueeze(dim=0)
        if batch is None:
            batch = img
        else:
            batch = torch.cat((batch, img),0)

    utils.imshow(torchvision.utils.make_grid(batch), save_path=save_path)

def load_embeddings(embeddings_dir, split):
    emb_matrix = pickle.load(open(os.path.join(embeddings_dir,f'emb_matrix_{split}.pt'), 'rb'))
    img_list = pickle.load(open(os.path.join(embeddings_dir, f'img_list_{split}.list'), 'rb'))
    return emb_matrix, img_list

def evaluate_query(embeddings_dir, attr_emb, splits, k=10):
    top_dists = []
    top_imgs = []
    for split in range(splits+1):
        # load embeddings for the current split
        emb_matrix, img_list = load_embeddings(embeddings_dir, split)

        topk_split = evaluate_query_split(attr_emb, emb_matrix,k)

        topk_split_dist = topk_split.values.view(-1).tolist()
        topk_split_indices = topk_split.indices.view(-1).tolist()
        topk_split_imgs = [img_list[k] for k in topk_split_indices]

        top_dists.extend(topk_split_dist)
        top_imgs.extend(topk_split_imgs)

    topk = np.argpartition(top_dists, k)[:k]
    topk_imgs = [top_imgs[idx] for idx in topk]
    return topk_imgs

# trian the model
if TRAIN_MODEL:
    img_enc, attr_enc = train_model(model_dir)
else:
    print(f"Loading weights from ./{model_dir}")
    imageEncoder.load_state_dict(torch.load(os.path.join(model_dir, f'img_encoder_900.pt')))
    attrEncoder.load_state_dict(torch.load(os.path.join(model_dir, f'attr_encoder_900.pt')))
    imageEncoder.eval()
    attrEncoder.eval()
    print(f"Weights Loaded!")


# Embed all the images
if SAVE_EMB:
    print('Extracting Embeddings for all images...')
    splits = embed_all_images(dataloader, imageEncoder, colors_dict, embeddings_dir)
    print(f'Embeddings saved in {splits} splits')
    meta_dict["SPLITS"] = splits
    pickle.dump(meta_dict, open(os.path.join(embeddings_dir, f'meta_dict.pkl'), 'wb'))


# Load the embeddings from disk
print('Loading embeddings from disk...')
meta_dict = pickle.load(open(os.path.join(embeddings_dir, f'meta_dict.pkl'), 'rb'))
splits = meta_dict["SPLITS"]
print(f"Found {splits} splits for embedding matrix")

# Enter the query

queries = random.sample(list(colors_dict.keys()),10)

# prev = None
for query in queries:
    # Embed the attribute
    
    attrEncoder = attrEncoder.to('cpu')
    attr_emb = attrEncoder(colors_dict[query])
    # if prev is None:
    #     prev = attr_emb
    #     print('prev is None')
    # else:
    #     torch.eq(attr_emb, prev)
    #     print('tensors are equal')
    # continue


    # compute topk according to the query
    print(f"Evaluating on query: {query}")
    topk_imgs = evaluate_query(embeddings_dir, attr_emb,  splits, k=3)
    # print(topk_imgs)
    # continue

    # plot top k images
    print("Plotting retrieved images...")
    plot_topk(topk_imgs, data_transforms, save_path=os.path.join(output_dir, query + '.png'))
