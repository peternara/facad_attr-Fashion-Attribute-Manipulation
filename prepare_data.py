import os
import json
import urllib.request
import tqdm
import pickle

file_name = 'meta_all_129927.json'
save_dir = '/data2fast/users/haroon/facad_data/'
images_dir = os.path.join(save_dir, 'images')

os.makedirs(save_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)


def read_data_json(file_name):
    # Read the json meta data file
    with open(file_name, 'r', encoding="utf-8") as f:
        data_list = json.load(f)

    return data_list

def get_dict_keys(data_dict):
    return data_dict[0].keys()

def get_category_dist(data_list):
    category_dict = {}
    for item in data_list:
        item_cat = item['category']

        if item_cat in category_dict:
            category_dict[item_cat] += 1
        else:
            category_dict[item_cat] = 1

    return category_dict

def get_topk(count_dict,THRESH=100):
    topk_cat = {k:count_dict[k] for k,v in count_dict.items() if v > THRESH}
    return topk_cat

def get_attribute_dist(data_dict):
    attr_dict = {}
    for item in data_dict:
        item_attr = item['attr']
        for attr in item_attr:
            if attr != 'nah':
                if attr in attr_dict:
                    attr_dict[attr] += 1
                else:
                    attr_dict[attr] = 1
    return attr_dict

def resolve_color(color_split, common_colors):
    for c in common_colors:
        if c in color_split:
            return c
    return None

def get_color_dist(data_dict, common_colors):
    color_dict = {}
    count = 0
    for item in data_dict:
        count += 1
        item_images = item['images']
        for image in item_images:
            color = image['color'].lower()
            color_split = color.split(' ')
            if len(color_split) > 1:
                color = resolve_color(color_split, common_colors)
                if color is None:
                    continue

            if color in color_dict:
                color_dict[color] += 1
            else:
                color_dict[color] = 0
        
    return color_dict

def filter_categories(data_list, topk_cat):
    cats = list(topk_cat.keys())
    new_data_list = []
    for item in data_list:
        item_cat = item['category']

        if item_cat in cats:
            new_data_list.append(item)
    return new_data_list

def filter_attributes(data_list, topk_att):
    attrs = list(topk_att.keys())
    new_data_list = []
    for item in data_list:
        item_attrs = item['attr']
        flag = False
        for attr in item_attrs:
            if attr in attrs:
                flag = True
                break
        if flag:
            new_data_list.append(item)
    return new_data_list

def initialize_dict(keys):
    d = {}
    for k in keys:
        d[k] = []
    return d

def resolve_attr(item_attr, attr_list):
    valid_attr = []
    for attr in item_attr:
        if attr in attr_list:
            valid_attr.append(attr)
    return valid_attr

def get_img_urls(img_dict):
    idx = 0
    image_list = []
    while True:
        try:
            image_url = img_dict[str(idx)]
            image_base_url = image_url.split("?")[0]
        except:
            break
        idx += 1
        image_list.append(image_base_url)
    return image_list

def download_images(img_names, images_to_download):
    assert len(img_names) == len(images_to_download)
    for img_name, img_url in zip(img_names, images_to_download):
        try:
            urllib.request.urlretrieve(img_url, img_name)
        except:
            return False
    return True

def write_list(item_list, filename):
    with open(filename, "w") as f:
        f.write("\n".join(item_list))


def download_data(file_name, save_dir, cat_list, attr_list, col_list):
    data_list = read_data_json(file_name)
    img_to_meta_dict = {}
    cat_img_dict = initialize_dict(cat_list)
    attr_img_dict = initialize_dict(attr_list)
    col_img_dict = initialize_dict(col_list)

    count = 0
    for item in tqdm.tqdm(data_list):
        item_cat = item['category']
        # if the cat is not in the list, continue
        if item_cat not in cat_list:
            continue
        item_attr = item['attr']
        valid_attr = resolve_attr(item_attr, attr_list)

        # if no valid attr is found, continue
        if len(valid_attr) == 0:
            continue

        item_id = item['id']
        images = item['images']

        for image in images:
            color = image['color'].lower()
            color_split = color.split(' ')
            if len(color_split) > 1:
                color = resolve_color(color_split, col_list)
                if color is None:
                    continue

            if color not in col_list:
                continue

            ## Download the images
            images_to_download = get_img_urls(image)
            img_names = [str(item_id) + '_' + str(color) + '_' + str(i) + '.jpg' for i in range(len(images_to_download))]
            img_paths = [os.path.join(save_dir, name) for name in img_names]
            success = download_images(img_paths, images_to_download)
            if not success:
                continue
            
            # update the category dict
            cat_img_dict[item_cat].extend(img_paths)

            # update the attribute dict
            for attr in valid_attr:
                attr_img_dict[attr].extend(img_paths)

            # update the color dict
            col_img_dict[color].extend(img_paths)
            attr_ids = [attr_list.index(attr) for attr in valid_attr]
            for img_name in img_names:
                meta_dict = {"category":item_cat,
                             "category_id":cat_list.index(item_cat),
                             "attr": valid_attr,
                             "attr_ids":attr_ids,
                             "color":color,
                             "color_id":color.index(color),
                             "title":item['title'],
                             "detail_info":item['detail_info'],
                             "description":item['description']}
                img_to_meta_dict[img_name] = meta_dict
        count += 1
    print(f"Download finished for {count} items ...")
    return img_to_meta_dict, cat_img_dict, attr_img_dict, col_img_dict


def dump_pickle(file_name, img_dict):
    with open(file_name, 'wb') as fp:
        pickle.dump(img_dict, fp)


common_colors = ['blue','green','red','purple','grey','yellow','black','white','brown','pink','gold','silver']

CAT_THRES = 1000
ATTR_THRES = 1000
COLOR_THRES = 1000

data_list = read_data_json(file_name)

category_dict = get_category_dist(data_list)
topk_cat = get_topk(category_dict, CAT_THRES)

print(f"Data List len before filtering cats: {len(data_list)}")
data_list = filter_categories(data_list, topk_cat)
print(f"Data List len after filtering cats: {len(data_list)}")

attr_dict = get_attribute_dist(data_list)
topk_att = get_topk(attr_dict, ATTR_THRES)

data_list = filter_attributes(data_list, topk_att)
print(f"Data List len after filtering attrs: {len(data_list)}")

color_dict = get_color_dist(data_list,common_colors)
topk_color = get_topk(color_dict, COLOR_THRES)



cat_list = list(topk_cat.keys())
attr_list = list(topk_att.keys())
col_list = list(topk_color.keys()) + ['purple']

attr_list = list(set(attr_list) - set(col_list))
print(f"Categories Selected: {len(cat_list)}")
print(f"Attributes Selected: {len(attr_list)}")
print(f"Colors Selected: {len(col_list)}\n")

# write files to disk
print("Writing category, attributes and color information to disk...")
write_list(cat_list, os.path.join(save_dir, 'category.txt'))
write_list(attr_list, os.path.join(save_dir, 'attributes.txt'))
write_list(col_list, os.path.join(save_dir, 'colors.txt'))

# Download data
print('Preparing to Download...')
img_to_meta_dict, cat_img_dict, attr_img_dict, col_img_dict = download_data(file_name, images_dir, cat_list, attr_list, col_list)

print('Download finished..')
print('Dumping Meta data to disk')
dump_pickle(os.path.join(save_dir, 'img_to_meta_dict.pkl'), img_to_meta_dict)
dump_pickle(os.path.join(save_dir,'cat_img_dict.pkl'), cat_img_dict)
dump_pickle(os.path.join(save_dir, 'attr_img_dict.pkl'), attr_img_dict)
dump_pickle(os.path.join(save_dir,'col_img_dict.pkl'), col_img_dict)

print('Finished...')




