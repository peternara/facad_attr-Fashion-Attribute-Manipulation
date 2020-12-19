import os
import numpy as np
import pickle
import random
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm
from PIL import Image

import utils
import datasets
import models

num_epochs = 2
ATTR_SIZE = 196
BATCH_SIZE = 8
SAVE_EMB = False
TRAIN_MODEL = False
CAL_SIMILARITY = False
TRAIN_RATIO = 0.85

data_dir = '/data2fast/users/haroon/facad_data/'
# data_dir = 'facad_data'

experiment_BASE = 'experiments'
experiment_name = os.path.join(experiment_BASE, 'triplet_base')
print(f"Running Experiment: {experiment_name} \n")
os.makedirs(experiment_name, exist_ok=True)

model_dir = os.path.join(experiment_name, 'weights')
embeddings_dir = os.path.join(experiment_name,'embeddings')
output_dir = os.path.join(experiment_name, 'output')

images_dir = os.path.join(data_dir, 'images')
imgs_list = os.listdir(images_dir)

# split data into train and test splits
if CAL_SIMILARITY:
    train_img_set, test_img_set = utils.train_test_split(experiment_name, imgs_list, TRAIN_RATIO)
    utils.dump_pickle(os.path.join(data_dir, 'train_img_set.pkl'), train_img_set)
    utils.dump_pickle(os.path.join(data_dir, 'test_img_set.pkl'), test_img_set)
else:
    train_img_set = utils.read_pickle(os.path.join(data_dir, 'train_img_set.pkl'))
    test_img_set = utils.read_pickle(os.path.join(data_dir, 'test_img_set.pkl'))

full_img_set = train_img_set + test_img_set

if CAL_SIMILARITY:
    print('Composing the attributes for all images..')
    img_attr_dict = utils.compose_img_attr_vects(data_dir, full_img_set)
    utils.dump_pickle(os.path.join(data_dir, 'img_attr_dict.pkl'), img_attr_dict)

    for mode in ['train', 'test']:
        print(f'Calculating the similarity scores for {mode} images...')
        if mode == 'train':
            similarity_dict = utils.cal_similarity_dict_torch(img_attr_dict, train_img_set)
            utils.dump_pickle(os.path.join(data_dir, 'train_similarity_dict.pkl'), similarity_dict)
        else:
            similarity_dict = utils.cal_similarity_dict_torch(img_attr_dict, test_img_set)
            utils.dump_pickle(os.path.join(data_dir, 'test_similarity_dict.pkl'), similarity_dict)

meta_dict = {}
train_embeddings_dir = os.path.join(embeddings_dir, 'train')
test_embeddings_dir = os.path.join(embeddings_dir, 'test')

os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(train_embeddings_dir, exist_ok=True)
os.makedirs(test_embeddings_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transforms = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])

train_dataset = datasets.FacadAttrDatasetTriplet(data_dir, train_img_set, mode='train',
                                        transforms = img_transforms)

test_dataset = datasets.FacadAttrDatasetTriplet(data_dir, test_img_set, mode='test',
                                        transforms = img_transforms)

train_dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        num_workers=BATCH_SIZE,
                        batch_size=BATCH_SIZE)

test_dataloader = DataLoader(test_dataset,
                        shuffle=False,
                        num_workers=BATCH_SIZE,
                        batch_size=BATCH_SIZE)

imageEncoder = models.ImageEncoder().to(device)
attrEncoder = models.AttrEncoder(ATTR_SIZE).to(device)

criterion = nn.TripletMarginLoss(margin=1.0, p=2)
params = list(imageEncoder.parameters()) + list(attrEncoder.parameters())

optimizer = optim.Adam(params,lr = 0.00005, weight_decay=0.001)


def train_model(BASE_PATH):
    iteration_number = 0
    counter = []
    loss_history = []
    train_accuracies = []
    test_accuracies = []
    test_precisions = []
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            anc_img, anc_attr ,pos_img, pos_attr, neg_img , neg_attr,_,_ = data
            anc_img, anc_attr, pos_img, pos_attr, neg_img, neg_attr = anc_img.to(device), anc_attr.to(device),pos_img.to(device), pos_attr.to(device), neg_img.to(device), neg_attr.to(device)

            diff_attrs = pos_attr - anc_attr
            optimizer.zero_grad()

            anc_emb = imageEncoder(anc_img)
            pos_emb = imageEncoder(pos_img)
            neg_emb = imageEncoder(neg_img)

            attr_emb = attrEncoder(diff_attrs.float())

            mse_loss = criterion(anc_emb + attr_emb, pos_emb, neg_emb)
            mse_loss.backward()
            optimizer.step()

            if i % 1000 == 0 :
                iteration_number +=1000
                counter.append(iteration_number)
                loss_history.append(mse_loss.item())
                print(f"Epoch {epoch}/{num_epochs} \t Current loss: {mse_loss.item()}\t Test Accs: {test_accuracies}\t P@10: {test_precisions}")

        if epoch % 1 == 0:
            print(f"\nCheckpointing at epoch : {epoch}")
            trian_save_dir = os.path.join(train_embeddings_dir, "epoch_" + str(epoch))
            test_save_dir = os.path.join(test_embeddings_dir, "epoch_" + str(epoch))

            os.makedirs(trian_save_dir, exist_ok=True)
            os.makedirs(test_save_dir, exist_ok=True)

            print("Embedding all images using latest model...")
            # train_splits = embed_all_images(train_dataloader, imageEncoder, os.path.join(train_embeddings_dir, "epoch_" + str(epoch)))
            test_splits = embed_all_images(test_dataloader, imageEncoder, test_save_dir)

            print('Calculating retrieval accuracy for current model...')
            # train_accuracy = test_model(train_dataloader, imageEncoder, attrEncoder, trian_save_dir, train_splits)
            test_accuracy, test_precision = test_model(test_dataloader, imageEncoder, attrEncoder, test_save_dir, test_splits)

            train_accuracy = 1.0

            train_accuracies.append(round(train_accuracy,4))
            test_accuracies.append(round(test_accuracy,4))
            test_precisions.append(round(test_precision, 4))
            print(f"Epoch {epoch+1}/{num_epochs} \t Current loss: {mse_loss.item()} \tTest Acc: {test_accuracy} \t P@10: {test_precision}")

            print("Saving model to disk..")
            torch.save(imageEncoder.state_dict(), os.path.join(BASE_PATH, f'img_encoder_{epoch}.pt'))
            torch.save(attrEncoder.state_dict(), os.path.join(BASE_PATH, f'attr_encoder_{epoch}.pt'))

    utils.plot_loss(loss_history, os.path.join(output_dir, 'loss.png'))
    utils.plot_accuracies(train_accuracies, test_accuracies, os.path.join(output_dir, 'acc.png'))
    print('Training Finished...')
    print(f"Train Accuracies: {train_accuracies}")
    print(f"Test Accuracies: {test_accuracies}")
    print(f"Precisions: {test_precisions}")
    return imageEncoder, attrEncoder


def embed_all_images(dataloader, model, embeddings_dir, mode='train'):
    model.eval()
    emb_matrix = None
    img_list = []
    split = 0
    for i, data in enumerate(dataloader):
        img, _ ,_ ,_,_,_, names,_ = data
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

def all_pairs_euclid_torch(A, B):
    sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
    sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()
    return torch.sqrt(
        sqrA - 2*torch.mm(A, B.t()) + sqrB
    )

def evaluate_query_split(attr_emb, emb_matrix, k=10):
    euclidean_distance = all_pairs_euclid_torch(attr_emb, emb_matrix)
    if euclidean_distance.shape[1] < k:
        k = euclidean_distance.shape[1]
    return torch.topk(euclidean_distance, k=k,largest=False,dim=1)

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
    batch_size = attr_emb.shape[0]
    top_dists = [[]]*batch_size
    top_imgs = [[]]*batch_size
    topk_imgs = []

    for split in range(splits+1):
        # load embeddings for the current split
        emb_matrix, img_list = load_embeddings(embeddings_dir, split)

        if emb_matrix is None:
            continue

        topk_split = evaluate_query_split(attr_emb, emb_matrix, k)

        topk_split_dist = topk_split.values.view(batch_size, -1).tolist()
        topk_split_indices = topk_split.indices.view(batch_size, -1).tolist()

        for i in range(batch_size):
            topk_split_imgs = [img_list[k] for k in topk_split_indices[i]]

            top_dists[i].extend(topk_split_dist[i])
            top_imgs[i].extend(topk_split_imgs)

    for i in range(batch_size):
        topk = np.argpartition(top_dists[i], k)[:k]
        topk_imgs.append([top_imgs[i][idx] for idx in topk])

    return topk_imgs

def eval_results(topk_predictions, targets, attr2, img_attr_dict):
    correct = 0
    total = 0
    true_positive = 0
    total_precision = 0
    correct_ids = []

    attr2 = attr2.cpu().numpy()
    for pred, target, t_attr in zip(topk_predictions, targets, attr2):
        total += 1
        Flag = True
        
        for img in pred:
            total_precision += 1
            img_attr = img_attr_dict[img]

            if np.allclose(img_attr,t_attr):
                if Flag:
                    correct_ids.append(total - 1)
                    correct += 1
                    Flag = False
                true_positive += 1
    return correct, total, true_positive, total_precision, correct_ids

def test_model(dataloader, imageEncoder, attrEncoder, embeddings_dir, splits):
    imageEncoder.eval()
    attrEncoder.eval()

    img_attr_dict = utils.read_pickle(os.path.join(data_dir, 'img_attr_dict.pkl'))


    total = 0
    correct = 0
    tp = 0
    count = 0
    prec_total = 0
    data_len = len(dataloader)
    for i, data in enumerate(dataloader):
        if i % 1000 == 0:
            print(f"Iteration: {i}/{data_len}")
        img1, attr1 ,img2, attr2,_,_, src_names, target_names = data
        img1, attr1, img2, attr2 = img1.to(device), attr1.to(device), img2.to(device), attr2.to(device)

        diff_attrs = attr2 - attr1

        add_attrs, sub_attrs = utils.decompose_feature_batch(data_dir, diff_attrs)

        img1_emb = imageEncoder(img1)
        attr_emb = attrEncoder(diff_attrs.float())

        topk = evaluate_query(embeddings_dir, img1_emb + attr_emb, splits)

        correct_batch, _, tp_batch, tp_total, correct_ids = eval_results(topk, target_names, attr2, img_attr_dict)

        if len(correct_ids) > 0:
            # utils.save_imgs(src_names, target_names, topk, correct_ids, images_dir,output_dir, img_attr_dict, add_attrs, sub_attrs)
            pass
        else:
            correct_ids = [4,2,6]
            utils.save_imgs(src_names, target_names, topk, correct_ids, images_dir,output_dir, img_attr_dict, add_attrs, sub_attrs)
            count += 1

        if count >= 10:
            print("Done!")
            assert False



        correct += correct_batch
        total += len(src_names)
        tp += tp_batch
        prec_total += tp_total

    return correct / total, tp / prec_total


# trian the model
if TRAIN_MODEL:
    imageEncoder, attrEncoder = train_model(model_dir)
else:
    print(f"Loading weights from ./{model_dir}")
    imageEncoder.load_state_dict(torch.load(os.path.join(model_dir, f'img_encoder_{num_epochs-1}.pt')))
    attrEncoder.load_state_dict(torch.load(os.path.join(model_dir, f'attr_encoder_{num_epochs-1}.pt')))
    imageEncoder.eval()
    attrEncoder.eval()
    print(f"Weights Loaded!")


# Embed all the images
if SAVE_EMB:
    print('Extracting Embeddings for test images...')
    splits = embed_all_images(test_dataloader, imageEncoder, embeddings_dir)
    print(f'Embeddings saved in {splits} splits')
    meta_dict["SPLITS"] = splits
    pickle.dump(meta_dict, open(os.path.join(embeddings_dir, f'meta_dict.pkl'), 'wb'))


# Load the embeddings from disk
print('Loading embeddings from disk...')
meta_dict = pickle.load(open(os.path.join(embeddings_dir, f'meta_dict.pkl'), 'rb'))
splits = meta_dict["SPLITS"]
print(f"Found {splits} splits for embedding matrix")

## Test the model
print("Evaluating Model...")
ret_accuracy, ret_prec = test_model(test_dataloader, imageEncoder, attrEncoder, embeddings_dir, splits)
print(f"Final Acc@10: {ret_accuracy * 100}\t P@10: {ret_prec * 100}")
