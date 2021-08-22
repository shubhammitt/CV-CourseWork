import cv2
import sys
import copy
import numpy as np
import itertools
import time


f_x1, f_y1, b_x1, b_y1, f_x2, f_y2, b_x2, b_y2 = -1, -1, -1, -1, -1, -1, -1, -1
drawing = False # true if mouse is pressed
ix, iy = -1, -1
count = 0
color_of_patch = (0, 255, 0)


def get_saliency_map(fg_map, bg_map):
	return [(i + (1 - j)) / 2 for i, j in zip(fg_map, bg_map)]


def get_likelihood_map(grayImage, x1, y1, x2, y2):
	x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
	rows, cols = grayImage.shape
	frequency = [0] * 256

	if x1 < 0: x1 = 0
	if x2 < 0: x2 = 0
	if y1 < 0: y1 = 0
	if y2 < 0: y2 = 0
	if x1 > cols: x1 = cols
	if x2 > cols: x2 = cols
	if y1 > rows: y1 = rows
	if y2 > rows: y2 = rows

	for i in range(y1, y2):
		for j in range(x1, x2):
			frequency[grayImage[i][j]] += 1

	max_frequency = max(frequency)
	if max_frequency != 0:
		frequency = [i / max_frequency for i in frequency]

	return frequency


def take_patches_from_user(originalImage):
	# mouse callback function
	def draw_2_patches(event, x, y, flags, param):

		global ix, iy, drawing, mode, count, color_of_patch, f_x1, f_y1, b_x1, b_y1, f_x2, f_y2, b_x2, b_y2

		if event == cv2.EVENT_LBUTTONDOWN:
			drawing = True
			ix, iy = x, y

		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
				cv2.rectangle(originalImage, (ix, iy), (x, y), color_of_patch, -1)  

		elif event == cv2.EVENT_LBUTTONUP:
			drawing = False
			cv2.rectangle(originalImage, (ix, iy), (x,y), color_of_patch,-1)

			count += 1

			if count == 1:
				f_x1, f_y1, f_x2, f_y2 = ix, iy, x, y
				color_of_patch = (0, 0, 255)
				print("Foreground patch done!\n")
				print("Make background patch...")
			else:
				b_x1, b_y1, b_x2, b_y2 = ix, iy, x, y
				print("Background patch done!")

	cv2.namedWindow('image')
	cv2.setMouseCallback('image', draw_2_patches)

	print("Make forground patch...")

	while 1:
	    cv2.imshow('image', originalImage)
	    cv2.waitKey(1) & 0xFF
	    if count == 2:
	        time.sleep(0.5)
	        break

	cv2.destroyAllWindows()
			

def main(image_path):
	originalImage = cv2.imread(image_path)
	take_patches_from_user(copy.deepcopy(originalImage))
	grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

	fg_map = get_likelihood_map(grayImage, f_x1, f_y1, f_x2, f_y2)
	bg_map = get_likelihood_map(grayImage, b_x1, b_y1, b_x2, b_y2)
	saliency_map = get_saliency_map(fg_map, bg_map)
	print("FG-map\n", fg_map)
	print("BG-map\n", bg_map)
	print("Salience map\n", saliency_map)
	saliency_map = [int(i * 255) for i in saliency_map]

	for i in range(grayImage.shape[0]):
		for j in range(grayImage.shape[1]):
			grayImage[i][j] = saliency_map[grayImage[i][j]]

	cv2.imshow('image', grayImage)
	cv2.waitKey()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 code.py <path_of_image>")
		exit(1)

	image_path = sys.argv[1]
	main(image_path)