import cv2
import sys
import copy
import numpy as np
import itertools

IMAGE_OF_SIZE = (240, 240)


def concatenate_feature_vectors(feature_vectors):
	'''
	Convert nested list to single list
	[ [...], [...], [...], ...] -> [... , ..., ... , .......]
	'''
	return list(itertools.chain(*feature_vectors))


def find_stats(matrix):
	'''
	find mean and standard deviation of matrix
	'''
	mean = matrix.mean()
	std = matrix.std()
	return [int(mean), int(std)]


def find_features_per_part(extracted_parts_of_image):
	'''
	for each part given in list, find 6 values for each part and return as nested list
	6 values -> [mean1, std1, mean2, std2, mean3, std3]
	'''
	feature_vectors = []

	for part in extracted_parts_of_image:
		vector = []
		for channel in range(3):
			matrix = part[:, :, channel] # take out chaanel i from part
			vector.extend(find_stats(matrix))

		feature_vectors.append(vector)

	return feature_vectors


def take_out_part(originalImage, x1, y1, horizontal_length, vertical_length):
	'''
	x1, y1 are starting coordinates in matrix and part is of given dimension
	'''
	return originalImage[x1: x1 + horizontal_length, y1 : y1 + vertical_length]


def divide_image_in_parts(originalImage):
	parts = [(2, 2), (3, 3), (4, 4), (5, 5)]
	extracted_parts_of_image = [] # stores image parts

	for x, y in parts:
		horizontal_length = IMAGE_OF_SIZE[0] // x
		vertical_length = IMAGE_OF_SIZE[1] // y

		for i in range(x):
			x1 = horizontal_length * i
			for j in range(y):
				y1 = vertical_length * j
				part = take_out_part(originalImage, x1, y1, horizontal_length, vertical_length)
				extracted_parts_of_image.append(part)
				# cv2.imshow('part', part)
				# cv2.waitKey()
				# cv2.destroyAllWindows()

	return extracted_parts_of_image


def main(image_path):
	originalImage = cv2.imread(image_path)
	originalImage = cv2.resize(originalImage, IMAGE_OF_SIZE) # resize the image for easy dividing

	extracted_parts_of_image = divide_image_in_parts(originalImage)
	feature_vectors = find_features_per_part(extracted_parts_of_image)
	global_feature_vector = concatenate_feature_vectors(feature_vectors)

	print(global_feature_vector)
	return global_feature_vector



if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 code.py <path_of_image>")
		exit(1)

	image_path = sys.argv[1]
	main(image_path)