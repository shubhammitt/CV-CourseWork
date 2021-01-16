# Otsu's algorithm

'''
Asssumptions: 
1) Object will be present at the center of the image.
2) Boundary pixels are likely to be the background. 
'''

import cv2
import sys
import copy


def get_frequency_distribution(grayImage):
	rows, cols = grayImage.shape

	frequency = [0] * 256

	for row in range(rows):
		for col in range(cols):
			frequency[grayImage[row][col]] += 1

	return frequency


def calculate_variance(frequency, start, end):
	'''
	end not inclusive
	'''
	mean = 0
	count = 0
	for i in range(start, end):
		mean += frequency[i] * i
		count += frequency[i]

	variance = 0
	for i in range(start, end):
		variance += ((mean - i) ** 2) * frequency[i]

	if count != 0:
		variance /= count
	return (count, variance)

def apply_assumption2(grayImage, min_threshold):
	'''
	Assumption: Boundary pixels are likely to be the background. 
	we will take frame of 10% on borders to be part of background
	we will find mean of pixel in that frame decide accordinly on which side of min_threashold that lies
	'''
	rows, cols = grayImage.shape

	left = int(0.1 * cols)
	right = int(0.9 * cols)
	up = int(0.1 * rows)
	down = int(0.9 * rows)

	sum_pixel = 0
	count = 0
	for row in range(rows):
		for col in range(cols):
			pixel = grayImage[row][col]
			if row <= up or row >= down or col <= left or col >= right:
				count += 1
				sum_pixel += pixel


	if count != 0:
		mean_pixel = sum_pixel / count

	print(min_threshold, mean_pixel)
	if mean_pixel <= min_threshold:
		return "left"

	return "right"


def extract_foreground(grayImage, originalImage, min_threshold):
	rows, cols = grayImage.shape
	print(rows, cols)
	blue_side = apply_assumption2(grayImage, min_threshold)

	for row in range(rows):
		for col in range(cols):
			if (grayImage[row][col] < min_threshold and blue_side == "left") or (grayImage[row][col] >= min_threshold and blue_side == "right"):
				originalImage[row][col][0] = 255 # blue
				originalImage[row][col][1] = 0 # green
				originalImage[row][col][2] = 0 # red

	cv2.imshow('gray', originalImage)
	cv2.waitKey()
	cv2.destroyAllWindows()


def otsu(originalImage):

	grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
	frequency = get_frequency_distribution(grayImage)
	
	min_threshold = 0
	min_variance = 0

	for i in range(1, 255):
		w0, v0 = calculate_variance(frequency, 0, i)
		w1, v1 = calculate_variance(frequency, i, 256)
		weighted_variance = w0 * v0 + w1 * v1

		if weighted_variance < min_variance or i == 1:
			min_variance = weighted_variance
			min_threshold = i

	extract_foreground(grayImage, originalImage, min_threshold)



if __name__ == '__main__':

	if len(sys.argv) != 2:
		print("Usage: python3 <script_name.py> <path_of_image>")
		exit(0)

	image_name = sys.argv[1]

	originalImage = cv2.imread(image_name) # original image is assumed to be colored

	otsu(copy.deepcopy(originalImage))
