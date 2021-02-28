import cv2
import sys
import numpy as np
import time
from multiprocessing import Pool

# length of cube 
SUB_CUBE_LEN = 4
rgb_frequency_dic = {}
pixel_frequency_list = []
saliency = {}

num_pixels = 0

def process_pixel(pixel1):
	'''
	for the pixel1, calculate the chebyshev distance
	'''
	pixel1_b, pixel1_g, pixel1_r = map(int, pixel1.split())

	chebyshev_distance = 0		
	for i in pixel_frequency_list:
		#  i = b, g, r, frequency of bgr
		chebyshev_distance += i[-1] * max(abs(pixel1_r - i[2]), \
						abs(pixel1_b - i[0]), abs(pixel1_g - i[1]))

	return (pixel1, chebyshev_distance)
		

def calculate_saliency(rgb_frequency_dic, originalImage):
	'''
		calculate saliency map using multiprocessing
	'''
	pool = Pool(4)	# Create a multiprocessing Pool with 4 workers
	pixel_saliency = pool.map(process_pixel, rgb_frequency_dic.keys())

	max_sal = 0

	# find maximum saliency in map
	for x, y in pixel_saliency:
		max_sal = max(max_sal, y)
	
	if max_sal == 0:
		max_sal = 1

	# normalise saliency map
	for x, y in pixel_saliency:
		saliency[x] = y / max_sal

	return saliency


def map_saliency(originalImage, saliency):
	'''
	convert every pixel to its saliency
	'''

	height, width = originalImage.shape[:2]

	image = [[0] * width for _ in range(height)]

	for i in range(height):
		for j in range(width):
			image[i][j] = saliency[str(originalImage[i][j][0]) + " " \
				+ str(originalImage[i][j][1]) + " " + str(originalImage[i][j][2])]
	
	image = np.float64(image)
	return image


def find_rgb_frequency(originalImage):
	'''
	find frequency of every pixel and store in dictionary
	'''
	global num_pixels
	height, width = originalImage.shape[:2]
	num_pixels = height * width

	# find which pixel belongs to which cube
	# and map it to median of that sub-cube
	map_pixel_to_cube = [0] * 256
	for i in range(256):
		map_pixel_to_cube[i] = i - i % SUB_CUBE_LEN + (SUB_CUBE_LEN >> 1)

	for i in range(height):
		for j in range(width):
			for k in range(3):
				originalImage[i][j][k] = map_pixel_to_cube[originalImage[i][j][k]]

			pixel = str(originalImage[i][j][0]) + " " + \
					str(originalImage[i][j][1]) + " " + str(originalImage[i][j][2])

			try:
				rgb_frequency_dic[pixel] += 1
			except:
				rgb_frequency_dic[pixel] = 1

	for i, j in rgb_frequency_dic.items():
		b, g, r = map(int, i.split())
		pixel_frequency_list.append([b, g, r , j])


def main(image_path):
	originalImage = cv2.imread(image_path)
	find_rgb_frequency(originalImage)
	calculate_saliency(rgb_frequency_dic, originalImage)
	final_image = map_saliency(originalImage, saliency)
	return final_image


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: python3 code.py <path_of_image> <SUB_CUBE_LEN>")
		exit(1)

	image_path = sys.argv[1]
	SUB_CUBE_LEN = int(sys.argv[2])

	# start timer
	start = time.time()
	final_image = main(image_path)
	# end timer
	end = time.time()
	print("Execution Time =", end - start,"sec")

	# display image
	cv2.imshow('saliency image', final_image)
	cv2.waitKey()
	cv2.destroyAllWindows()

