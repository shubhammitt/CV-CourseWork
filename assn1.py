import cv2
import sys
import copy
from queue import Queue
from math import *


white = 255
black = 0
object_color = white
background_color = black


def color_to_BW(color_image, threshold=127):
	'''
	converts colored image to black and white
	'''
	grayImage = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
	thresh, bw_image = cv2.threshold(grayImage, threshold, white, cv2.THRESH_BINARY)
	return bw_image


def is_valid_move(x, y, rows, cols):
	'''
	check validity of cell whether it is inside matrix or not
	'''
	return x < rows and y < cols and x >= 0 and y >= 0


def color_the_component(row, col, color, bw_image):
	'''
	color the given component with color using BFS
	'''
	rows, cols = len(bw_image), len(bw_image[0])
	x_min, y_min = col, row
	x_max, y_max = col, row
	x_y = set([(col, row)])

	q = Queue()

	q.put((row, col))
	bw_image[row][col] = color # color current pixel

	moves = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] # 8 possible nearby pixels
	while q.qsize() != 0:

		x1, y1 = q.get()

		# check nearby pixels
		for move in moves:
			x2, y2 = x1 + move[0], y1 + move[1]

			if is_valid_move(x2, y2, rows, cols) and (bw_image[x2][y2] == object_color):
				# nearby pixel is valid and is of white color then color that nearby pixel
				q.put((x2, y2)) # keep nearby pixels in the queue for further recursive coloring
				
				x_min = min(x_min, y2)
				x_max = max(x_max, y2)
				y_min = min(y_min, x2)
				y_max = max(y_max, x2)
				bw_image[x2][y2] = color # color nearby pixel

				for _move in moves:
					x3, y3 = x2 + _move[0], y2 + _move[1]
					if (is_valid_move(x3, y3, rows, cols)) and bw_image[x3][y3] == background_color:
						x_y.add((y2, x2))

	center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
	perimeter_x, perimeter_y = 0, 0
	search_length = 4
	min_radius = 0

	for center_x in range(max(0, center[0] - search_length), min(cols, center[0] + search_length)):
		for center_y in range(max(0, center[1] - search_length), min(rows, center[1] + search_length)):
			max_radius = 0
			local_perimeter_x, local_perimeter_y = 0, 0
			for x, y in x_y:
				r = (x - center_x)**2 + (y - center_y)**2
				if r > max_radius:
					max_radius = r
					local_perimeter_x, local_perimeter_y = x, y

			if min_radius == 0 or min_radius > max_radius:
				min_radius = max_radius
				perimeter_x, perimeter_y = local_perimeter_x, local_perimeter_y
				center = (center_x, center_y)

	return (center, (perimeter_x, perimeter_y), ceil(min_radius**0.5))


def find_diameter_cors_connected_components(bw_image):
	rows, cols = bw_image.shape
	bw_image = bw_image.tolist() # numpy matrix to normal list of lists
	diameters_coordinates = {}

	color = 0

	for row in range(rows):
		for col in range(cols):
			if bw_image[row][col] == object_color:
				color -= 1
				diameters_coordinates[abs(color)] = color_the_component(row, col, color, bw_image)

	return diameters_coordinates



def main(image_path):

	originalImage = cv2.imread(image_path)
	bw_image = color_to_BW(originalImage) # convert color to BW image
	diameters_coordinates = find_diameter_cors_connected_components(bw_image)

	font=cv2.FONT_HERSHEY_PLAIN 

	# Actual bounding circle
	# img = originalImage
	# contours,hierarchy = cv2.findContours(bw_image, 4, 5)
	# for i in range(len(contours)):
	# 	cnt = contours[i]
	# 	(x,y),radius = cv2.minEnclosingCircle(cnt)
	# 	center = (int(x),int(y))
	# 	radius = int(radius)
	# 	img = cv2.circle(img,center,radius,(0,255,0),2)
	# cv2.imshow('Black white image', img)
	# cv2.waitKey()
	
	for i in diameters_coordinates:
		center, perimeter, radius = diameters_coordinates[i]
		if radius > 30 and 1.5 * radius < originalImage.shape[0] and 1.5 * radius < originalImage.shape[1]:
			print("Center =", center, "Radius =", radius)
			originalImage = cv2.circle(originalImage, center, radius, (0, 0, 255), 2)

			cv2.putText(originalImage, str(center), center, font, 1.2, (88, 12, 255), 2)
			cv2.line(originalImage ,center, perimeter, (211, 2, 252), 3)
	cv2.imshow('Ojects detected', originalImage)
	cv2.waitKey()
	cv2.destroyAllWindows()



if __name__ == "__main__":

	if len(sys.argv) != 2:
		print("Usage: python3 assn1.py <path_of_image>")
		exit(1)

	image_path = sys.argv[1]
	main(image_path)
