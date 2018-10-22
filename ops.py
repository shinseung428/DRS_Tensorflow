# encoding: utf-8
from glob import glob 
import os
import tensorflow as tf
import numpy as np
import cv2
import json
import sys
import random 
from scipy.io import loadmat
from scipy.spatial import distance as dist
import math

FLAGS = tf.app.flags.FLAGS

#function to save images in tile
#comment this function block if you don't have opencv
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

	cv2.imwrite(FLAGS.res_path+"/img_"+str(epoch) + ".jpg", (tile_img + 1)*127.5)


