import os
import json
import urllib.request
import tqdm

file_name = 'meta_all_129927.json'
save_dir = 'data'
os.makedirs(save_dir, exist_ok=True)

with open(file_name, 'r', encoding="utf-8") as f:
    data_list = json.load(f)

dict_keys = data_list[0].keys()

total = 0
total_items = 100
for item in tqdm.tqdm(data_list):
	total += 1
	if total == total_items:
		print("Generated 100 data points")
		assert False
	
	item_id = item['id'] 
	item_path = os.path.join(save_dir, str(item_id))
	if os.path.exists(item_path):
		print(f'{item_id} already exists')
		continue

	os.makedirs(item_path, exist_ok=True)

	images_list = item['images']
	
	for images in images_list:
		idx = 0

		item_color = images['color'].replace('/',' ')
		# color_path = os.path.join(item_path, item_color)
		# os.makedirs(color_path, exist_ok=True)
		
		while True:
			try:
				image_url = images[str(idx)]
				image_base_url = image_url.split("?")[0]
				img_name = str(item_id) + "_" + item_color + "_" + str(idx) + '.jpg'
				urllib.request.urlretrieve(image_base_url, os.path.join(item_path, img_name))
			except:
				break
			
			idx += 1

