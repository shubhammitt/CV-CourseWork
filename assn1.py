import cv2
import sys
import copy
from queue import Queue


white = 255
black = 0
object_color = white


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
	x, y = [col], [row]

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
				x.append(y2)
				y.append(x2)
				x_min = min(x_min, y2)
				x_max = max(x_max, y2)
				y_min = min(y_min, x2)
				y_max = max(y_max, x2)
				bw_image[x2][y2] = color # color nearby pixel

	return (x_min, y_min, x_max, y_max, x, y)


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
		x1, y1, x2, y2, x, y = diameters_coordinates[i]
		center = ((x1 + x2) // 2, (y1 + y2) // 2)
		radius = int(((max([((p - center[0])** 2 + (q- center[1]) ** 2) for p, q in zip(x, y)]))**0.5))
		originalImage = cv2.circle(originalImage, center, radius, (0, 0, 255), 2)

	cv2.imshow('Ojects detected', originalImage)
	cv2.waitKey()
	cv2.destroyAllWindows()



if __name__ == "__main__":

	if len(sys.argv) != 2:
		print("Usage: python3 assn1.py <path_of_image>")
		exit(1)

	image_path = sys.argv[1]
	main(image_path)
