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
	'''
	returns a mapping of pixel values with their frequency in the grayImage of original Inamge
	'''
	rows, cols = grayImage.shape

	frequency = [0] * 256 # index denotes the pixel values so frequency[i] denotes the count of pixels of value i in the grayImage

	for row in range(rows):
		for col in range(cols):
			frequency[grayImage[row][col]] += 1

	return frequency


def calculate_variance(frequency, start, end):
	'''
	'end' not inclusive in the range
	'''
	mean = 0
	count = 0 # number of pixel in given range

	for i in range(start, end):
		mean += frequency[i] * i
		count += frequency[i]

	variance = 0
	for i in range(start, end):
		variance += ((mean - i) ** 2) * frequency[i]

	if count != 0:
		variance /= count
	return (count, variance)


def apply_assumption1(grayImage, min_threshold):
	'''
	Assumption: Object will be present at the center of the image.
	1) Considere the rectangle whose length and width are ⅕ of given image dimensions and its center coincides with that of the given image’s center.
	2) Find the median of the pixels which are present in that rectangle. Let it is denoted by the median_pixel.
	3) If median_pixel lies to the left of the min_threshold then the left side is foreground otherwise the right side is foreground
	Note: This function will return which side is background.
	'''

	# print("Using Assumption 1")
	rows, cols = grayImage.shape

	center_x, center_y = rows // 2, cols // 2

	# length and breadth are 20% of the original image 
	l_box = int(0.2 * cols)
	b_box = int(0.2 * rows)

	#(x, y) denotes the top left coordinate of the rectangle we are considering
	x = center_x - b_box // 2
	y = center_y - l_box // 2

	count = 0 # count number of pixels in the sub-rectangle 
	arr = [] # store the pixel values in the sub-rectangle

	for i in range(x, x + l_box):
		for j in range(y, y + b_box):
			arr.append(grayImage[i][j])
			count += 1

	arr.sort()
	median_pixel = arr[count // 2]

	if median_pixel <= min_threshold:
		return "right" # we are returning which side will be background

	return "left"


def apply_assumption2(grayImage, min_threshold):
	'''
	Assumption: Boundary pixels are likely to be the background.
	1) We will take frame of 10% on borders to be part of background
	2) We will find median of pixel in that frame decide accordingly on which side of min_threashold that lies
	3) If median_pixel lies to the left of the min_threshold then the left side is background otherwise the right side is the background.
	'''
	# print("Using Assumption 2")
	rows, cols = grayImage.shape

	left = int(0.1 * cols)
	right = int(0.9 * cols)
	up = int(0.1 * rows)
	down = int(0.9 * rows)

	count = 0 # count number of pixels in the border part 
	arr = [] # store the pixel values in the border

	for row in range(rows):
		for col in range(cols):
			pixel = grayImage[row][col]
			if row <= up or row >= down or col <= left or col >= right: # if cell lies in the border part
				count += 1
				arr.append(pixel)

	arr.sort()
	median_pixel = arr[count // 2]

	if median_pixel <= min_threshold:
		return "left"

	return "right"


def extract_foreground(grayImage, originalImage, min_threshold, assumption):
	rows, cols = grayImage.shape

	# choose whohc assumption to apply
	if assumption == 2:
		background_side = apply_assumption2(grayImage, min_threshold)
	else:
		background_side = apply_assumption1(grayImage, min_threshold)

	for row in range(rows):
		for col in range(cols):
			if (grayImage[row][col] <= min_threshold and background_side == "left") or (grayImage[row][col] >= min_threshold and background_side == "right"):
				originalImage[row][col][0] = 255 # blue
				originalImage[row][col][1] = 0 # green
				originalImage[row][col][2] = 0 # red

	# cv2.imshow('gray', originalImage)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	return originalImage


def otsu(originalImage, grayImage, assumption, min_threshold=None):
	'''
	Apply Otsu Algorithm on given Image
	'''

	if min_threshold is not None:
		return extract_foreground(grayImage, originalImage, min_threshold, assumption)

	frequency = get_frequency_distribution(grayImage) # a mapping of pixel values with their frequency in the grayImage of original Inamge
	
	min_threshold = 0
	min_variance = 10**100

	for i in range(1, 255):
		w0, v0 = calculate_variance(frequency, 0, i)
		w1, v1 = calculate_variance(frequency, i, 256)
		weighted_variance = w0 * v0 + w1 * v1

		if weighted_variance < min_variance or i == 1:
			min_variance = weighted_variance
			min_threshold = i

			
	print("Minimum Threshold =", min_threshold)
	return extract_foreground(grayImage, originalImage, min_threshold, assumption)
