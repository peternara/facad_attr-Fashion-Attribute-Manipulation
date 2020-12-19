import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.metrics import pairwise_distances
import tqdm
import random
import shutil
import itertools
import textwrap

import torch
import torchvision
import torch.nn.functional as F
import cv2

def imshow(img,text=None,should_save=False, save_path=None):
    inp = img.numpy()
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.axis("off")
    if text:
        plt.text(50, 4, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

    if save_path is not None:
        plt.imsave(save_path, inp)

    plt.imshow(inp)
    plt.show()

def get_similarity_scores(src, topk, img_attr_dict):

    src_attr = img_attr_dict[src].reshape(1, -1)

    scores = []
    for img in topk:
        target_attr = img_attr_dict[img].reshape(1, -1)
        score = 1 - pairwise_distances(src_attr, target_attr, metric="cosine")

        scores.append((img, round(score[0][0], 2)))
    return scores


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def write_text_on_image(img, text):
    img_h, img_w, _ = img.shape
    wrapped_text = textwrap.wrap(text, width=img_w)

    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    font_size = 1
    font_thickness = 1

    x, y = 0, img_h
    
    for i, line in enumerate(wrapped_text):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 1

        y = int((img_h + textsize[1]) // 2) + i * gap
        x = int((img_w - textsize[0]) // 2)

        cv2.putText(img, line, (x, y), font,
                    font_size, 
                    (0,0,255), 
                    font_thickness, 
                    lineType = cv2.LINE_AA)
    return img

def save_imgs(src_names, target_names, topk, correct_ids, imgs_dir, output_dir, img_attr_dict, add_attrs, sub_attrs):
    margin = 2 #Margin between pictures in pixels
    w = 6 # Width of the matrix (nb of images)
    h = 2 # Height of the matrix (nb of images)
    n = w*h

    src_names = [src_names[i] for i in correct_ids]
    target_names = [target_names[i] for i in correct_ids]
    topk = [topk[i] for i in correct_ids]
    add_attrs = [add_attrs[i] for i in correct_ids]
    sub_attrs = [sub_attrs[i] for i in correct_ids]

    for i, image in enumerate(src_names):
        add = add_attrs[i]
        sub = sub_attrs[i]
        

        scores = get_similarity_scores(target_names[i], topk[i], img_attr_dict)
        scores.sort(key = lambda x: x[1], reverse=True)

        filename_list = [image, target_names[i]] + [x[0] for x in scores]
        scores = ["",target_names[i]] + [x[1] for x in scores]

        imgs = np.array([cv2.resize(cv2.imread(imgs_dir+"/"+file), (480, 720)) for file in filename_list])

        #Define the shape of the image to be replicated (all images should have the same shape)
        img_h, img_w, img_c = imgs[0].shape

        border = np.zeros((int(img_h*0.2), img_w, 3))
        border_h, border_w, _ = border.shape

        img_h = img_h + border_h

        #Define the margins in x and y directions
        m_x = margin
        m_y = margin

        #Size of the full size image
        mat_x = img_w * w + m_x * (w - 1)
        mat_y = img_h * h + m_y * (h - 1)

        #Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
        imgmatrix = np.zeros((mat_y, mat_x, img_c),np.uint8)
        imgmatrix.fill(255)

        #Prepare an iterable with the right dimensions
        positions = itertools.product(range(h), range(w))
        font = cv2.FONT_HERSHEY_SIMPLEX

        i = 0
        for (y_i, x_i), img in zip(positions, imgs):
            x = x_i * (img_w + m_x)
            y = y_i * (img_h + m_y)

            img = np.concatenate((img, border), axis = 0)

            if i == 0:
                cv2.putText(img, "Add: " + str(add), (0, img_h - 100), font, 0.5, (255, 255, 255), 1)
                cv2.putText(img, "Sub: " + str(sub), (0, img_h - 50), font, 0.5, (255, 255, 255), 1)
            elif i == 1:
                cv2.putText(img, str(scores[i]), (0, img_h - 100), font, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(img, str(scores[i]), (img_w // 2 - 50, img_h - 50), font, 2, (255, 255, 255), 2)
            imgmatrix[y:y+img_h, x:x+img_w, :] = img
            i += 1

        # resized = cv2.resize(imgmatrix, (mat_x//6,mat_y//6), interpolation = cv2.INTER_AREA)
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        cv2.imwrite(os.path.join(output_dir, image), imgmatrix, compression_params)


def read_list_from_file(file_name):
    with open(file_name, 'r') as f:
        data_list = f.readlines()

    data_list = [x.strip("\n") for x in data_list]
    return data_list

def read_pickle(file_path):
    return pickle.load(open(file_path, 'rb'))

def dump_pickle(file_name, img_dict):
    with open(file_name, 'wb') as fp:
        pickle.dump(img_dict, fp)

def all_pairs_euclid_torch(A, B):
    sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
    sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()
    return torch.sqrt(
        sqrA - 2*torch.mm(A, B.t()) + sqrB
    )


def compose_feature(img_dict, colors_list):

    cat = img_dict['category_id']
    attrs = img_dict['attr_ids']
    col = colors_list.index(img_dict['color'])

    num_cats = 30
    num_attrs = 152
    num_colors = 14

    cat_vect = np.zeros(num_cats)
    attr_vector = np.zeros(num_attrs)
    col_vect = np.zeros(num_colors)

    cat_vect[cat] = 1

    for attr in attrs:
        attr_vector[attr] = 1

    col_vect[col] = 1

    return np.concatenate([cat_vect, attr_vector, col_vect])

def decompose_feature(data_dir, feature):

    cat_list = read_list_from_file(os.path.join(data_dir, 'category.txt'))
    attr_list = read_list_from_file(os.path.join(data_dir, 'attributes.txt'))
    colors_list = read_list_from_file(os.path.join(data_dir, 'colors.txt'))

    feature_list = cat_list + attr_list + colors_list

    num_cats = 30
    num_attrs = 152
    num_colors = 14

    add_attrs = torch.nonzero((feature == 1), as_tuple=False)
    sub_attrs = torch.nonzero((feature == -1), as_tuple=False)

    add_attrs = [feature_list[i] for i in add_attrs]
    sub_attrs = [feature_list[i] for i in sub_attrs]

    return add_attrs, sub_attrs

def decompose_feature_batch(data_dir, feature_batch):
    add_attrs = []
    sub_attrs = []

    for feature in feature_batch:
        add, sub = decompose_feature(data_dir, feature)
        add_attrs.append(add)
        sub_attrs.append(sub)
    return add_attrs, sub_attrs


def compose_img_attr_vects(data_dir, imgs_list):
    img_to_meta_dict = read_pickle(os.path.join(data_dir, 'img_to_meta_dict.pkl'))
    colors_list = read_list_from_file(os.path.join(data_dir, 'colors.txt'))

    img_attr_dict = {}
    count = 0
    for img in imgs_list:
        try:
            img_dict = img_to_meta_dict[img]
        except:
            count += 1
            print(img)
            print(count)
            continue
        
        feature_vect = compose_feature(img_dict, colors_list)

        img_attr_dict[img] = feature_vect
    
    return img_attr_dict


def cal_similarity_dict(img_attr_dict):
    imgs = sorted(img_attr_dict.keys())
    similarity_dict = {}
    for i in tqdm.tqdm(range(len(imgs))):
        similarity_list = []
        for j in range(len(imgs)):
            if i != j:
                similarity_score = 1 - spatial.distance.cosine(img_attr_dict[imgs[i]], img_attr_dict[imgs[j]])
                if similarity_score == 1:
                    continue
                similarity_list.append((imgs[j], similarity_score))
        similarity_list.sort(key = lambda x: x[1], reverse=True) 
        similarity_dict[imgs[i]] = similarity_list
    return similarity_dict

def cal_similarity_dict_vect(img_attr_dict):
    imgs = sorted(img_attr_dict.keys())

    attr_matrix = np.array([img_attr_dict[img] for img in imgs], dtype="int8")
    similarity_matrix = 1 - pairwise_distances(attr_matrix, metric="cosine")
    topk_similar = torch.topk(torch.tensor(similarity_matrix), k=20,dim=1)

    topk_scores = topk_similar.values
    topk_indices = topk_similar.indices

    similarity_dict = {}
    for i in tqdm.tqdm(range(len(imgs))):
        scores_i, indices_i = topk_scores[i], topk_indices[i]

        similarity_list = [(imgs[idx], score.item()) for idx, score in zip(indices_i, scores_i) if score < 0.99]
        similarity_dict[imgs[i]] = similarity_list
    return similarity_dict

def cal_similarity_dict_torch(img_attr_dict, imgs):
    similarity_dict = {}
    attr_matrix = torch.tensor([img_attr_dict[img] for img in imgs])


    start = 0
    jump = 1000
    end = start + jump
    while start < len(attr_matrix):
        subset_attr_matrix = attr_matrix[start:end]

        similarity_matrix = 1 - pairwise_distances(subset_attr_matrix, attr_matrix, metric="cosine")
        
        topk_similar = torch.topk(torch.tensor(similarity_matrix), k=10,dim=1)  


        topk_scores = topk_similar.values
        topk_indices = topk_similar.indices

        if start % 10000 == 0:
            print(f"Processing Img: {start}")

        for i in range(start, end):
            scores_i, indices_i = topk_scores[i - start], topk_indices[i - start]
            similarity_list = [(imgs[idx], score.item()) for idx, score in zip(indices_i, scores_i)]
            similarity_dict[imgs[i]] = similarity_list

        start = end
        end = start + jump

        if end > len(attr_matrix):
            end = len(attr_matrix)
    return similarity_dict

def train_test_split(save_dir, imgs_list, TRAIN_RATIO):
    items = {img.split('_')[0] for img in imgs_list}
    train_items = random.sample(items, int(TRAIN_RATIO * len(items)))
    test_items = list(set(items) - set(train_items))

    print(f"Adding {len(train_items)} items to train set..")
    print(f"Adding {len(test_items)} items to test set..")

    dump_pickle(os.path.join(save_dir, 'train_items.pkl'), train_items)
    dump_pickle(os.path.join(save_dir, 'test_items.pkl'), test_items)

    train_set = []
    test_set = []

    for img in imgs_list:
        img_item = img.split('_')[0]
        if img_item in test_items:
            test_set.append(img)
        else:
            train_set.append(img)

    return train_set, test_set

def organize_imgs_into_folders(data_dir):
    imgs_dir = os.path.join(data_dir, 'images')
    new_dir = os.path.join(data_dir,'imgs')
    os.makedirs(new_dir, exist_ok=True)

    imgs_list = os.listdir(imgs_dir)

    print(f"Organizing {len(imgs_list)} images into folders")
    for img in tqdm.tqdm(imgs_list):
        folder_name = img.split('_')[0]
        folder_path = os.path.join(new_dir, folder_name)
        os.makedirs(os.path.join(folder_path), exist_ok=True)
        src_img = os.path.join(imgs_dir, img)
        shutil.copy(src_img, folder_path)

def folder_list_to_img_list(data_dir, folder_list):
    imgs_list = []

    for folder in folder_list:
        folder_path = os.path.join(data_dir, folder)
        imgs = os.listdir(folder_path)
        imgs_list.extend(imgs)

    return imgs_list

def subtract_vects(vect1, vect2):
    return vect2 - vect1

def plot_loss(loss_history, save_path):
    plt.plot(range(len(loss_history)),loss_history)
    plt.title("Training Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(save_path)

def plot_accuracies(list1, list2, save_path):
    labels = ['epoch_' + str(i*10) for i in range(len(list1))]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, list1, width, label='Train')
    rects2 = ax.bar(x + width/2, list2, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Train and Test Accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig(save_path)




