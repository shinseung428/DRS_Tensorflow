from glob import glob 
import cv2
import numpy as np




def img_tile(epoch, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
	if imgs.ndim != 3 and imgs.ndim != 4:
		raise ValueError('imgs has wrong number of dimensions.')
	n_imgs = imgs.shape[0]

	tile_shape = None
	# Grid shape
	img_shape = np.array(imgs.shape[1:3])
	if tile_shape is None:
		img_aspect_ratio = img_shape[1] / float(img_shape[0])
		aspect_ratio *= img_aspect_ratio
		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
		grid_shape = np.array((tile_height, tile_width))
	else:
		assert len(tile_shape) == 2
		grid_shape = np.array(tile_shape)

	# Tile image shape
	tile_img_shape = np.array(imgs.shape[1:])
	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

	# Assemble tile image
	tile_img = np.empty(tile_img_shape)
	tile_img[:] = border_color
	for i in range(grid_shape[0]):
		for j in range(grid_shape[1]):
			img_idx = j + i*grid_shape[1]
			if img_idx >= n_imgs:
				# No more images - stop filling out the grid.
				break
			img = imgs[img_idx]
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			yoff = (img_shape[0] + border) * i
			xoff = (img_shape[1] + border) * j
			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

	cv2.imwrite("./results/img_"+str(epoch) + ".jpg", tile_img)


all_pths = glob('./sampled_results/*.jpg')

processed = {}
for idx, pth in enumerate(all_pths):
	new_path = pth.split('_')[-1][:-4]
	if processed.get(pth) == None:
		processed[pth] = float(new_path)

sorted_by_value = sorted(processed.items(), key=lambda kv: kv[1])

bucket = []
counter = 0
for elem in reversed(sorted_by_value):
	path = elem[0]
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	bucket.append(img)
	if len(bucket) == 64:
		img_tile(counter, np.array(bucket))
		bucket = []
		counter += 1 
		print ("saved")
# print (sorted_by_value)
