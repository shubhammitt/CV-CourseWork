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
	grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
	thresh, bw_image = cv2.threshold(grayImage, threshold, white, cv2.THRESH_BINARY)
	return bw_image


def get_BW_image(originalImage, is_image_bw):
	if is_image_bw != "1" or len(originalImage.shape) != 2:
		bw_image = color_to_BW(originalImage)
	else:
		bw_image = originalImage

	return bw_image


def is_valid_move(x, y, rows, cols):
	'''
	check validity of cell inside matrix
	'''
	return x < rows and y < cols and x >= 0 and y >= 0


def color_the_component(row, col, color, bw_image):
	'''
	color the given component with color
	'''
	rows, cols = len(bw_image), len(bw_image[0])
	q = Queue()
	q.put((row, col))
	bw_image[row][col] = color

	moves = [(0, 1), (1, 0), (-1, 0), (0, -1)]

	while q.qsize() != 0:
		x1, y1 = q.get()

		for move in moves:
			x2, y2 = x1 + move[0], y1 + move[1]

			if is_valid_move(x2, y2, rows, cols) and (bw_image[x2][y2] == object_color):
				q.put((x2, y2))
				bw_image[x2][y2] = color


def count_connected_components(bw_image):
	rows, cols = bw_image.shape
	bw_image = bw_image.tolist()

	color = 0

	for row in range(rows):
		for col in range(cols):
			if bw_image[row][col] == object_color:
				color -= 1
				color_the_component(row, col, color, bw_image)

	print("Number of Objects in given image =", abs(color))



if __name__ == "__main__":

	if len(sys.argv) != 3:
		'''
		is_image_bw should be "1" if original image is already black and white
		'''
		print("Usage: python3 <script_name.py> <path_of_image> <is_image_bw?>")
		exit(1)

	image_name = sys.argv[1]
	is_image_bw = sys.argv[2]

	originalImage = cv2.imread(image_name)
	bw_image = get_BW_image(originalImage, is_image_bw)

	# apply algo
	count_connected_components(copy.deepcopy(bw_image))

	cv2.imshow('Black white image', bw_image)
	cv2.waitKey()
	cv2.destroyAllWindows()
