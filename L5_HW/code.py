import cv2
import sys
import copy
import numpy as np
import itertools

def concatenate_histograms(feature_vectors):
	'''
	Convert nested list to single list
	[ [...], [...], [...], ...] -> [... , ..., ... , .......]
	'''
	return list(itertools.chain(*feature_vectors))


def parity(a, b):
	if a >= b :
		return 1
	return 0


def find_histogram_of_patch(extracted_patches_of_image):
	histograms = []

	for patch in extracted_patches_of_image:
		frequency = [0] * 256

		for i in range(patch.shape[0]):
			for j in range(patch.shape[1]):
				# patch[i][j] --> concerned pixel

				decimal = 0
				directions = [(-1, -1), (-1, 0), (-1, 1), (0 , 1), (1, 1), (1, 0), (1, -1), (0, -1)]

				for x, y in directions:
					try:
						z = parity(patch[i + x, j + y], patch[i][j])
					except:
						# this will excecute when concerned pixel is at boundary of patch
						z = parity(0, patch[i][j])
					decimal = 2 * decimal + z

				frequency[decimal] += 1
				
		histograms.append(frequency)

	return histograms


def take_out_patch(grayImage, x1, y1, horizontal_length, vertical_length):
	'''
	x1, y1 are starting coordinates in matrix and part is of given dimension
	'''
	return grayImage[x1: x1 + horizontal_length, y1 : y1 + vertical_length]


def divide_image_in_patches(grayImage):
	parts = [(2, 2)]
	extracted_patches_of_image = [] # stores image parts

	IMG_SIZE = grayImage.shape
	for x, y in parts:
		horizontal_length = IMG_SIZE[0] // x
		vertical_length = IMG_SIZE[1] // y

		for i in range(x):
			x1 = horizontal_length * i
			for j in range(y):
				y1 = vertical_length * j
				patch = take_out_patch(grayImage, x1, y1, horizontal_length, vertical_length)
				extracted_patches_of_image.append(patch)

	return extracted_patches_of_image


def main(image_path):

	originalImage = cv2.imread(image_path)
	grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

	extracted_patches_of_image = divide_image_in_patches(grayImage)

	histograms = find_histogram_of_patch(extracted_patches_of_image)

	feature_vectors = concatenate_histograms(histograms)

	print(feature_vectors)
	return feature_vectors



if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 code.py <path_of_image>")
		exit(1)

	image_path = sys.argv[1]
	main(image_path)