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


def apply_assumtion1(grayImage, min_threshold):
	'''
	'''
	rows, cols = grayImage.shape

	center_x, center_y = rows // 2, cols // 2

	l_box = int(0.2 * cols)
	b_box = int(0.2 * rows)

	x = center_x - b_box // 2
	y = center_y - l_box // 2
	count = 0
	arr = []

	for i in range(x, x + l_box):
		for j in range(y, y + b_box):
			arr.append(grayImage[i][j])
			count += 1

	arr.sort()
	median_pixel = arr[count // 2]

	if median_pixel <= min_threshold:
		return "left"

	return "right"


def apply_assumption2(grayImage, min_threshold):
	'''
	Assumption: Boundary pixels are likely to be the background.
	we will take frame of 10% on borders to be part of background
	we will find median of pixel in that frame decide accordinly on which side of min_threashold that lies
	'''
	rows, cols = grayImage.shape

	left = int(0.1 * cols)
	right = int(0.9 * cols)
	up = int(0.1 * rows)
	down = int(0.9 * rows)

	count = 0
	arr = []
	for row in range(rows):
		for col in range(cols):
			pixel = grayImage[row][col]
			if row <= up or row >= down or col <= left or col >= right:
				count += 1
				arr.append(pixel)

	arr.sort()
	median_pixel = arr[count // 2]

	if median_pixel <= min_threshold:
		return "left"

	return "right"


def extract_foreground(grayImage, originalImage, min_threshold, assumption):
	rows, cols = grayImage.shape
	print(rows, cols)

	if assumption == 2:
		blue_side = apply_assumption2(grayImage, min_threshold)
	else:
		blue_side = apply_assumtion1(grayImage. min_threshold)

	for row in range(rows):
		for col in range(cols):
			if (grayImage[row][col] < min_threshold and blue_side == "left") or (grayImage[row][col] >= min_threshold and blue_side == "right"):
				originalImage[row][col][0] = 255 # blue
				originalImage[row][col][1] = 0 # green
				originalImage[row][col][2] = 0 # red

	cv2.imshow('gray', originalImage)
	cv2.waitKey()
	cv2.destroyAllWindows()


def otsu(originalImage, assumption):

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

	extract_foreground(grayImage, originalImage, min_threshold, assumption)



if __name__ == '__main__':

	if len(sys.argv) != 3:
		print("Usage: python3 <script_name.py> <path_of_image> <assumption>")
		exit(0)

	image_name = sys.argv[1]
	assumption = max(2, int(sys.argv[2]))

	originalImage = cv2.imread(image_name) # original image is assumed to be colored

	otsu(copy.deepcopy(originalImage), assumption)
