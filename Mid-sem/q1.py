# Otsu's algorithm for TSS

'''
Asssumptions: Object will be present at the center of the image.
'''

import cv2
import sys


def apply_assumption(grayImage, min_threshold):
	'''
	Assumption: Object will be present at the center of the image.
	1) Considere the rectangle whose length and width are ⅕ of given image dimensions and its center coincides with that of the given image’s center.
	2) Find the median of the pixels which are present in that rectangle. Let it is denoted by the median_pixel.
	3) If median_pixel lies to the left of the min_threshold then the left side is foreground otherwise the right side is foreground
	Note: This function will return which side is background.
	'''

	print("Using Assumption: Object will be present at the center of the image")
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


def extract_foreground(grayImage, originalImage, min_threshold):
	'''
	This will make the background of blue color according to optimal threshhold returned by otse
	'''
	rows, cols = grayImage.shape
	
	background_side = apply_assumption(grayImage, min_threshold)

	for row in range(rows):
		for col in range(cols):
			if (grayImage[row][col] <= min_threshold and background_side == "left") or (grayImage[row][col] >= min_threshold and background_side == "right"):
				originalImage[row][col][0] = 255 # blue
				originalImage[row][col][1] = 0 # green
				originalImage[row][col][2] = 0 # red

	cv2.imshow('Otsu\'s Output', originalImage)
	cv2.waitKey()
	cv2.destroyAllWindows()


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


def calculate_tss(frequency, start, end):
	'''
	'end' not inclusive in the range
	'''
	mean = 0
	count = 0 # number of pixel in given range

	for i in range(start, end):
		mean += frequency[i] * i
		count += frequency[i]

	if count != 0:
		mean /= count

	tss = 0
	for i in range(start, end):
		tss += frequency[i] * ((mean - i)**2)

	return tss


def otsu(grayImage, originalImage):
	'''
	Apply Otsu Algorithm on given Image
	Class 1: [0, minthreshold)
	Class 2: [minthreashold, 256]
	'''

	frequency = get_frequency_distribution(grayImage) # a mapping of pixel values with their frequency in the grayImage of original Inamge
	
	min_threshold = 0
	min_tss = 0

	for i in range(1, 255):
		tss0 = calculate_tss(frequency, 0, i)
		tss1 = calculate_tss(frequency, i, 256)
		tss =  tss0 + tss1

		if tss <= min_tss or i == 1:
			min_tss = tss
			min_threshold = i
		
	print("Optimal Threshold =", min_threshold)
	return min_threshold


def main(image_name):
	originalImage = cv2.imread(image_name)
	grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
	min_threshold = otsu(grayImage, originalImage)
	extract_foreground(grayImage, originalImage, min_threshold)


if __name__ == "__main__":

	if len(sys.argv) != 2:
		print("Usage: python3 <script_name.py> <path_of_image>")
		exit(1)

	image_name = sys.argv[1]
	main(image_name)
