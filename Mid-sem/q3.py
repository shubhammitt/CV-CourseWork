import cv2
import sys
import copy
import numpy as np
import itertools, math
import time
from multiprocessing import Pool

rgb_frequency = {}
saliency = {}

num_pixels = 0

def process_pixel(pixel1):
	pixel1_b, pixel1_g, pixel1_r = map(int, pixel1.split())
	chebyshev_distance = 0		
	for pixel2, frequency2 in rgb_frequency.items():
		pixel2_b, pixel2_g, pixel2_r = map(int, pixel2.split())
		chebyshev_distance += frequency2 * max(abs(pixel1_r - pixel2_r), abs(pixel1_b - pixel2_b), abs(pixel1_g - pixel2_g))

	return (pixel1, chebyshev_distance)
		

def calculate_saliency(rgb_frequency, originalImage):
	global saliency
	pool = Pool(4)                         # Create a multiprocessing Pool
	pixel_saliency = pool.map(process_pixel, rgb_frequency.keys())
	max_sal = 0
	for x, y in pixel_saliency:
		max_sal = max(max_sal, y)
	
	for x, y in pixel_saliency:
		saliency[x] = y / max_sal
	return saliency


def map_saliency(originalImage, saliency):
	height, width = originalImage.shape[:2]
	image = [[0] * width for _ in range(height)]
	for i in range(height):
		for j in range(width):
			image[i][j] = saliency[str(originalImage[i][j][0]) + " " + str(originalImage[i][j][1]) + " " + str(originalImage[i][j][2])]
	
	image=np.float64(image)
	# cv2.imshow('saliency', image)
	# cv2.waitKey()
	# cv2.destroyAllWindows()


def find_rgb_frequency(originalImage):
	global rgb_frequency, num_pixels
	height, width = originalImage.shape[:2]
	num_pixels = height * width
	rgb_frequency = {}
	for i in range(height):
		for j in range(width):
			pixel = str(originalImage[i][j][0]) + " " + str(originalImage[i][j][1]) + " " + str(originalImage[i][j][2])
			try:
				rgb_frequency[pixel] += 1
			except:
				rgb_frequency[pixel] = 1

	return rgb_frequency


def main(image_path):
	originalImage = cv2.imread(image_path)
	# originalImage = cv2.resize(originalImage, (100, 100))
	rgb_frequency = find_rgb_frequency(originalImage)
	saliency = calculate_saliency(rgb_frequency, originalImage)
	map_saliency(originalImage, saliency)



if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 code.py <path_of_image>")
		exit(1)

	image_path = sys.argv[1]
	main(image_path)
	