import os
import json
import urllib.request
import tqdm
import pickle
import operator

import matplotlib.pyplot as plt

def plot_data(data_dict, k =10):
	D = dict(sorted(data_dict.items(), key=operator.itemgetter(1), reverse=True)[:k])
	plt.bar(*zip(*D.items()))
	plt.show()

def get_category_counts(data_list):
	category_dict = {}
	for item in tqdm.tqdm(data_list):
		item_category = item['category']
		if item_category in category_dict:
			category_dict[item_category] += 1
		else:
			category_dict[item_category] = 0

	return category_dict




file_name = 'meta_all_129927.json'
save_dir = 'polo'
os.makedirs(save_dir, exist_ok=True)

with open(file_name, 'r', encoding="utf-8") as f:
    data_list = json.load(f)

dict_keys = data_list[0].keys()
print(dict_keys)

# plot_data(get_category_counts(data_list), k=5)

total = 0
for item in tqdm.tqdm(data_list):
	
	item_id = item['id'] 
	item_category = item['category']
	
	# only get the images for the desired category
	if item_category != save_dir:
		continue

	total += 1
	item_path = os.path.join(save_dir, str(item_id))
	os.makedirs(item_path, exist_ok=True)

	images_list = item['images']
	for images in images_list:
		idx = 0

		item_color = images['color'].replace('/',' ')
		
		while True:
			try:
				image_url = images[str(idx)]
				image_base_url = image_url.split('?')[0]
				img_name = str(item_id) + "_" + item_color + "_" + str(idx) + '.jpg'
				urllib.request.urlretrieve(image_base_url, os.path.join(item_path, img_name))
			except:
				break
			
			idx += 1

