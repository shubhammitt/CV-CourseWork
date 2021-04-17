import cv2
import sys

from queue import Queue
import copy
import numpy as np
import itertools
from fcmeans import FCM
import pandas as pd
from matplotlib import pyplot as plt

def perform_fcm_clustering(dataset, number_of_clusters):
	fcm = FCM(n_clusters=number_of_clusters)
	fcm.fit(dataset)
	return fcm

def get_labels(fcm, dataset, height, width):
	labels = fcm.predict(dataset)
	return labels.reshape(height, width).tolist()


def is_valid_move(x, y, rows, cols):
	'''
	check validity of cell whether it is inside matrix or not
	'''
	return x < rows and y < cols and x >= 0 and y >= 0

def color_the_component(row, col, color, matrix, labels, rows, cols):

	# store all coordinates which are at perimeter of current object
	surrounding_labels = {}
	q = Queue()
	number_of_pixels = 1
	my_label = labels[row][col]


	q.put((row, col))
	 # color current pixel
	matrix[row][col] = color

	# Note: 8 nearby pixels are possible 
	# but we are taking only in 4 directions 
	# as single diagonal of object will not be possible to see with naked eyes
	moves = [(0, 1), (1, 0), (-1, 0), (0, -1)] 
	
	while q.qsize() != 0:

		x1, y1 = q.get()
		# print(x1,y1)

		# check nearby pixels
		for move in moves:
			x2, y2 = x1 + move[0], y1 + move[1]

			if is_valid_move(x2, y2, rows, cols):
				# print(x2,y2,"points")	
				# print(labels[x2][y2], my_label, "labels")
				# print("color", matrix[x2][y2])
				if (labels[x2][y2] == my_label) :
					if matrix[x2][y2] == 0:
						# print("color it")
						# nearby pixel is valid and is of same label as given
						
						q.put((x2, y2)) # keep nearby pixels in the queue for further recursive coloring
						matrix[x2][y2] = color # color nearby pixel
						number_of_pixels += 1
				else:
					other_label = int(labels[x2][y2])
					if other_label not in surrounding_labels:
						surrounding_labels[other_label] = 0

					surrounding_labels[other_label] += 1

	# find most surrounding labels
	m = max(surrounding_labels.values())
	most_frequent_label_in_surrounding = [i for i in surrounding_labels if surrounding_labels[i] == m][0]
	return (number_of_pixels, most_frequent_label_in_surrounding)

def merge_small(matrix, info_of_colors, labels, rows, cols):

	for row in range(rows):
		for col in range(cols):
			if matrix[row][col] in info_of_colors:
				labels[row][col] = info_of_colors[matrix[row][col]]


def find_connected_components_and_merge_small(rows, cols, labels, Threshold):
	'''
	find all connected components formed by objects in a binary image
	'''

	matrix = np.zeros((rows, cols), dtype=np.int32)
	matrix = matrix.tolist() # numpy matrix to normal list of lists for fast access
	info_of_colors = {}
	color = 0

	for row in range(rows):
		for col in range(cols):
			if matrix[row][col] == 0:
				color -= 1
				number_of_pixels, most_frequent_label_in_surrounding = color_the_component(row, col, color, matrix, labels, rows, cols)
				if number_of_pixels < Threshold:
					info_of_colors[color] = most_frequent_label_in_surrounding

	merge_small(matrix, info_of_colors, labels, rows, cols)


def main(image_path, number_of_clusters, Threshold):

	originalImage = cv2.imread(image_path)
	height, width = originalImage.shape[:2]

	dataset = np.array([np.append(np.array([originalImage[i][j][k] / 255 for k in range(3)]), [i / height, j / width], 0) for i in range(height) for j in range(width)])
	fcm = perform_fcm_clustering(dataset, number_of_clusters)
	labels = get_labels(fcm, dataset, height, width)
	centers = (fcm.centers).tolist()

	for row in range(height):
		for col in range(width):
			label = labels[row][col]
			originalImage[row, col] = [int(centers[label][0] * 255), int(255 * centers[label][1]), int(255* centers[label][2])] 

	cv2.imshow('Image before merging', originalImage)
	cv2.waitKey()
	cv2.destroyAllWindows()
	
	find_connected_components_and_merge_small(height, width, labels, Threshold)

	for row in range(height):
		for col in range(width):
			label = labels[row][col]
			originalImage[row, col] = [int(centers[label][0] * 255), int(255 * centers[label][1]), int(255* centers[label][2])] 

	cv2.imshow('Image after merging', originalImage)
	cv2.waitKey()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print("Usage: python3 code.py <path_of_image> <number_of_clusters> <Threshold>")
		exit(1)

	image_path = sys.argv[1]
	number_of_clusters = int(sys.argv[2])
	Threshold = int(sys.argv[3])
	main(image_path, number_of_clusters, Threshold)