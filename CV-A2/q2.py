import cv2
import sys
import copy
import numpy as np
import itertools, math
import time
import matplotlib.pyplot as plt

def slic(originalImage):
	slic = cv2.ximgproc.createSuperpixelSLIC(originalImage ,region_size=16, ruler = 12.0) 
	slic.iterate(20)     #Number of iterations, the greater the better
	return slic

def find_superpixels_centers_rgb(slic_img, originalImage):
	num_superpixel = slic_img.getNumberOfSuperpixels()
	labels = slic_img.getLabels()
	superpixel_centers_rgb = [[0]*4 for _ in range(num_superpixel)]

	height, width, channel = originalImage.shape

	for i in range(height):
		for j in range(width):
			cluster_label = labels[i][j]
			superpixel_centers_rgb[cluster_label][3] += 1
			for k in range(channel):
				superpixel_centers_rgb[cluster_label][k] += originalImage[i][j][k]

	for i in range(num_superpixel):
		for j in range(3):
			superpixel_centers_rgb[i][j] /= superpixel_centers_rgb[i][3]
		superpixel_centers_rgb[i] = superpixel_centers_rgb[i][:3]
	return superpixel_centers_rgb

def find_superpixels_centers_xy(slic_img, originalImage):
	num_superpixel = slic_img.getNumberOfSuperpixels()
	labels = slic_img.getLabels()
	superpixel_centers = [[0]*3 for _ in range(num_superpixel)]

	height, width, channel = originalImage.shape

	for i in range(height):
		for j in range(width):
			cluster_label = labels[i][j]
			superpixel_centers[cluster_label][2] += 1
			superpixel_centers[cluster_label][0] += i
			superpixel_centers[cluster_label][1] += j

	for i in range(num_superpixel):
		for j in range(2):
			superpixel_centers[i][j] /= superpixel_centers[i][2]
		superpixel_centers[i] = superpixel_centers[i][:2]
	return superpixel_centers


def get_cluster_of_superpixels(superpixel_centers_rgb, num_clusters):

	Z = np.array(superpixel_centers_rgb)
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret, cluster_labels, cluster_centers = cv2.kmeans(Z, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	cluster_centers = np.uint8(cluster_centers)

	# cv2.imshow('res2',res2)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return cluster_centers, cluster_labels

def norm(a, b):
	return sum([(i - j)**2 for i, j in zip(a, b)]) ** 0.5

def contrast_cue(num_clusters, label, cluster_centers):
	cluster_centers = np.int32(cluster_centers)
	N = label.shape[0]
	contrast_cue_list = []
	frequency_pixels = [0] * (num_clusters)

	for i in label:
		frequency_pixels[i[0]] += 1

	for k in range(num_clusters):
		cue_k = 0
		for i in range(num_clusters):
			if i != k:
				cue_k += frequency_pixels[i] * norm(cluster_centers[k], cluster_centers[i])

		contrast_cue_list.append(cue_k / N)

	return contrast_cue_list


def spatial_cue(num_clusters, label, center, superpixel_centers_xy):
	n = len(superpixel_centers_xy)
	spatial_cue_list = []
	pixel_label = [[] for _ in range(num_clusters)]
	sigma = 100
	frequency_pixels = [0] * (num_clusters)
	center_of_superpixels = [sum(i[0] for i in superpixel_centers_xy) / n,  sum(i[1] for i in superpixel_centers_xy) / n]

	for i in label:
		frequency_pixels[i[0]] += 1

	for i in range(n):
		pixel_label[label[i][0]].append(superpixel_centers_xy[i])

	for k in range(num_clusters):
		cue = 0
		for pixel_loc in pixel_label[k]:
			cue += math.exp(-(norm(center_of_superpixels ,pixel_loc)/sigma))
		spatial_cue_list.append(cue / frequency_pixels[k])

	return spatial_cue_list

def map_cue_to_image(cue, num_clusters, originalImage, superpixel_labels, cluster_labels):
	# normalise cue
	max_cue = max(cue)
	cue = [i / max_cue for i in cue]
	superpixels_to_cue = {}
	image = [[0] * originalImage.shape[1] for _ in range(originalImage.shape[0])]

	for i in range(len(cluster_labels)):
		superpixels_to_cue[i] = cue[cluster_labels[i][0]]

	for i in range(originalImage.shape[0]):
		for j in range(originalImage.shape[1]):
			image[i][j] = np.float(superpixels_to_cue[superpixel_labels[i][j]])

	image=np.float32(image)
	cv2.imshow('cue image', image)
	cv2.waitKey()
	cv2.destroyAllWindows()

def main(image_path, num_clusters):
	originalImage = cv2.imread(image_path)
	slic_img = slic(originalImage)
	superpixel_labels = slic_img.getLabels()

	superpixel_centers_rgb = find_superpixels_centers_rgb(slic_img, originalImage)

	cluster_centers, cluster_labels = get_cluster_of_superpixels(superpixel_centers_rgb, num_clusters)

	contrast_cue_list = contrast_cue(num_clusters, cluster_labels, cluster_centers)

	superpixel_centers_xy = find_superpixels_centers_xy(slic_img, originalImage)
	spatial_cue_list = spatial_cue(num_clusters, cluster_labels, cluster_centers, superpixel_centers_xy)

	
	map_cue_to_image(contrast_cue_list, num_clusters, originalImage, superpixel_labels, cluster_labels)
	map_cue_to_image(spatial_cue_list, num_clusters, originalImage, superpixel_labels, cluster_labels)


	# print(mask)
if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: python3 code.py <path_of_image> <num_clusters>")
		exit(1)

	image_path = sys.argv[1]
	num_clusters = int(sys.argv[2])
	main(image_path, num_clusters)