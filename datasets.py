import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import utils

class PoloColorDataset(Dataset):
    
    def __init__(self,data_dir, transform=None):
        self.imageFolderDataset = dset.ImageFolder(root=data_dir)
        self.colors_dict = self.get_colors_dict()  
        self.transform = transform

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

    def get_colors_dict(self):
        colors_dict = {}
        for img in self.imageFolderDataset.imgs:
            color = img[0].split('/')[-1].split('_')[1].lower()
            if color not in colors_dict:
                colors_dict[color] = torch.tensor([len(colors_dict)], dtype=torch.long)
        return colors_dict
        
    def __getitem__(self,index):
        img_tuple = self.imageFolderDataset.imgs[index]

        color = img_tuple[0].split('/')[-1].split('_')[1].lower()
        pos_color = self.colors_dict[color]

        img = Image.open(img_tuple[0])
        if self.transform is not None:
            img = self.transform(img)

        neg_color = random.choice(list(set(self.colors_dict.keys()) - set([color])))
        neg_color = self.colors_dict[neg_color]
        
        return img, pos_color, neg_color,  img_tuple[0]


class FacadAttrDataset(Dataset):

    def __init__(self, data_dir, imgs_list, mode,transforms=None):
        self.images_dir = os.path.join(data_dir, 'images')

        self.images = imgs_list
        self.colors = utils.read_list_from_file(os.path.join(data_dir, 'colors.txt'))
        self.attrs = utils.read_list_from_file(os.path.join(data_dir, 'attributes.txt'))
        self.cats = utils.read_list_from_file(os.path.join(data_dir, 'category.txt'))

        self.img_to_meta_dict = utils.read_pickle(os.path.join(data_dir, 'img_to_meta_dict.pkl'))
        self.col_img_dict = utils.read_pickle(os.path.join(data_dir, 'col_img_dict.pkl'))
        self.attr_img_dict = utils.read_pickle(os.path.join(data_dir, 'attr_img_dict.pkl'))
        self.cat_img_dict = utils.read_pickle(os.path.join(data_dir, 'cat_img_dict.pkl'))

        if mode == 'train':
            self.similarity_dict = utils.read_pickle(os.path.join(data_dir, 'train_similarity_dict.pkl'))
        else:
            self.similarity_dict = utils.read_pickle(os.path.join(data_dir, 'test_similarity_dict.pkl'))

        self.transforms = transforms


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        source_img = self.images[index]
        similar_imgs = self.similarity_dict[source_img][1:10]

        # print(len(similar_imgs))
        target_img = random.choice(similar_imgs)[0]

        # get the meta-data information related to source img dict
        source_img_dict = self.img_to_meta_dict[source_img]
        source_attr_vect = torch.tensor(utils.compose_feature(source_img_dict, self.colors))

        # get the meta-data information related to source img dict
        target_img_dict = self.img_to_meta_dict[target_img]
        target_attr_vect = torch.tensor(utils.compose_feature(target_img_dict, self.colors))

        # read image and apply transforms
        source_img_pil = Image.open(os.path.join(self.images_dir, source_img)).convert('RGB')
        target_img_pil = Image.open(os.path.join(self.images_dir, target_img)).convert('RGB')

        if self.transforms is not None:

            source_img_tensor = self.transforms(source_img_pil)
            target_img_tensor = self.transforms(target_img_pil)

        # print(source_img_tensor.shape, source_attr_vect.shape, target_img_tensor.shape, target_attr_vect.shape, source_img, target_img)
        return source_img_tensor, source_attr_vect, target_img_tensor, target_attr_vect, source_img, target_img


class FacadAttrDatasetTriplet(Dataset):

    def __init__(self, data_dir, imgs_list, mode,transforms=None):
        self.images_dir = os.path.join(data_dir, 'images')

        self.images = imgs_list
        self.colors = utils.read_list_from_file(os.path.join(data_dir, 'colors.txt'))
        self.attrs = utils.read_list_from_file(os.path.join(data_dir, 'attributes.txt'))
        self.cats = utils.read_list_from_file(os.path.join(data_dir, 'category.txt'))

        self.img_to_meta_dict = utils.read_pickle(os.path.join(data_dir, 'img_to_meta_dict.pkl'))
        self.col_img_dict = utils.read_pickle(os.path.join(data_dir, 'col_img_dict.pkl'))
        self.attr_img_dict = utils.read_pickle(os.path.join(data_dir, 'attr_img_dict.pkl'))
        self.cat_img_dict = utils.read_pickle(os.path.join(data_dir, 'cat_img_dict.pkl'))

        if mode == 'train':
            self.similarity_dict = utils.read_pickle(os.path.join(data_dir, 'train_similarity_dict.pkl'))
        else:
            self.similarity_dict = utils.read_pickle(os.path.join(data_dir, 'test_similarity_dict.pkl'))

        self.transforms = transforms


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        source_img = self.images[index]
        similar_imgs = self.similarity_dict[source_img][1:10]

        # print(len(similar_imgs))
        target_img = random.choice(similar_imgs)[0]

        negative_samples = self.similarity_dict[target_img][1:10]
        negative_img = random.choice(negative_samples)[0]

        # get the meta-data information related to source img dict
        source_img_dict = self.img_to_meta_dict[source_img]
        source_attr_vect = torch.tensor(utils.compose_feature(source_img_dict, self.colors))

        # get the meta-data information related to target img dict
        target_img_dict = self.img_to_meta_dict[target_img]
        target_attr_vect = torch.tensor(utils.compose_feature(target_img_dict, self.colors))

        # get the meta-data information related to negative img dict
        negative_img_dict = self.img_to_meta_dict[negative_img]
        negative_attr_vect = torch.tensor(utils.compose_feature(negative_img_dict, self.colors))

        # read image and apply transforms
        source_img_pil = Image.open(os.path.join(self.images_dir, source_img)).convert('RGB')
        target_img_pil = Image.open(os.path.join(self.images_dir, target_img)).convert('RGB')
        negative_img_pil = Image.open(os.path.join(self.images_dir, negative_img)).convert('RGB')

        if self.transforms is not None:

            source_img_tensor = self.transforms(source_img_pil)
            target_img_tensor = self.transforms(target_img_pil)
            negative_img_tensor = self.transforms(negative_img_pil)

        # print(source_img_tensor.shape, source_attr_vect.shape, target_img_tensor.shape, target_attr_vect.shape, source_img, target_img)
        return source_img_tensor, source_attr_vect, target_img_tensor, target_attr_vect, negative_img_tensor, negative_attr_vect, source_img, target_img
