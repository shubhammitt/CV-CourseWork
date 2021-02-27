import cv2
import sys
import copy
import numpy as np
import itertools

def min_max_ratio_to_bin(a, b):
	a += 0.00001
	b += 0.00001
	return round(min(a, b) / max(a, b))


def find_lbp_feature_map(grayImage):
	height, width = grayImage.shape

	image = np.zeros(grayImage.shape, dtype = np.uint8)

	for i in range(height):
		for j in range(width):

			decimal = 0
			directions = [(-1, -1), (-1, 0), (-1, 1), (0 , 1), (1, 1), (1, 0), (1, -1), (0, -1)]

			for x, y in directions:
				try:
					z = min_max_ratio_to_bin(grayImage[i + x, j + y], grayImage[i][j])
				except:
					# this will excecute when concerned pixel is at boundary of patch
					z = 1
				decimal = 2 * decimal + z

			image[i][j] = decimal

	cv2.imshow('lbp', image)
	cv2.waitKey()
	cv2.destroyAllWindows()

				
	

def main(image_path):

	originalImage = cv2.imread(image_path)
	grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

	find_lbp_feature_map(grayImage)
	



if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 code.py <path_of_image>")
		exit(1)

	image_path = sys.argv[1]
	main(image_path)