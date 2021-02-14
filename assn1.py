import cv2
import sys
from queue import Queue
from math import *

white = 255
black = 0
object_color = white
background_color = black
bounding_circle_info = {}
search_length = 5


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
	main work: color the given component with color using BFS
	side work: find coordinates which are at perimeter of object 
			   and diagonal coordinates of bounding rectangle
	'''
	rows, cols = len(bw_image), len(bw_image[0])
	# below corrdinates are diagonals of bounding rectangle
	x_min, y_min = col, row
	x_max, y_max = col, row
	# store all coordinates which are at perimeter of current object
	border_x_y = set([(col, row)]) 
	num_white_pixels = 1
	q = Queue()

	q.put((row, col))
	 # color current pixel
	bw_image[row][col] = color

	# Note: 8 nearby pixels are possible 
	# but we are taking only in 4 directions 
	# as single diagonal of object will not be possible to see with naked eyes
	moves = [(0, 1), (1, 0), (-1, 0), (0, -1)] 
	
	while q.qsize() != 0:

		x1, y1 = q.get()

		# check nearby pixels
		for move in moves:
			x2, y2 = x1 + move[0], y1 + move[1]

			if is_valid_move(x2, y2, rows, cols) and (bw_image[x2][y2] == object_color):
				# nearby pixel is valid and is of white color then color that nearby pixel
				
				q.put((x2, y2)) # keep nearby pixels in the queue for further recursive coloring
				bw_image[x2][y2] = color # color nearby pixel
				num_white_pixels += 1
				

				# find whether (x2, y2) are at perimeter of current object
				for _move in moves:
					x3, y3 = x2 + _move[0], y2 + _move[1]
					if not is_valid_move(x3, y3, rows, cols) or bw_image[x3][y3] == background_color:
						border_x_y.add((y2, x2))
						# find diagonal points of bounding rectangle
						x_min = min(x_min, y2)
						x_max = max(x_max, y2)
						y_min = min(y_min, x2)
						y_max = max(y_max, x2)
						break

	find_bounding_circle_info(x_min, x_max, y_min, y_max, border_x_y, rows, cols, color, num_white_pixels)


def find_bounding_circle_info(x_min, x_max, y_min, y_max, border_x_y, rows, cols, color, num_white_pixels):
	'''
	This will find center, farthest point from center and radius of circle 
	and store it in bounding_circle_info with its corresponding color
	'''
	center = ((x_min + x_max) >> 1, (y_min + y_max) >> 1)
	farthest_point = (0, 0)
	min_radius = 10**18

	for center_x in range(max(0, center[0] - search_length), min(cols, center[0] + search_length)):
		for center_y in range(max(0, center[1] - search_length), min(rows, center[1] + search_length)):
			
			# find radius (center_x, center_y) are assumed to be center of bounding circle
			max_radius = 0
			local_farthest_point = (0, 0)
			for x, y in border_x_y:
				r = (x - center_x) ** 2 + (y - center_y) ** 2

				if r > max_radius:
					max_radius = r
					local_farthest_point = (x, y)

					if max_radius >= min_radius:
						break

			if min_radius > max_radius:
				min_radius = max_radius
				farthest_point = local_farthest_point
				center = (center_x, center_y)

	bounding_circle_info[abs(color)] = (center, farthest_point, ceil(min_radius**0.5), num_white_pixels)


def find_connected_components(bw_image):
	'''
	find all connected components formed by objects in a binary image
	'''
	rows, cols = bw_image.shape
	bw_image = bw_image.tolist() # numpy matrix to normal list of lists for fast access

	color = 0

	for row in range(rows):
		for col in range(cols):
			if bw_image[row][col] == object_color:
				color -= 1
				color_the_component(row, col, color, bw_image)


def make_bounding_circle(originalImage):
	rows, cols = originalImage.shape[:2]
	for i in bounding_circle_info:
		center, farthest_point, radius, num_white_pixels = bounding_circle_info[i]
		# ignore too small objects
		if radius > 10:
			jaccard_similarity = find_jaccard_similarity(rows, cols, center, radius, num_white_pixels)
			print("Center =", center, "\tRadius =", radius, "\tJaccard Similarity =", jaccard_similarity)
			
			originalImage = cv2.circle(originalImage, center, radius, (0, 255, 1), 2) # draw circle
			cv2.putText(originalImage, str(center), center, cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 12, 255), 2) # put cordinates of center
			cv2.line(originalImage ,center, farthest_point, (255, 21, 25), 2) # make a line from center to farthest point

	cv2.imshow('Ojects detected', originalImage)
	cv2.waitKey()
	cv2.destroyAllWindows()


def find_jaccard_similarity(rows, cols, center, radius, num_white_pixels):
	num_pixels_in_circle = 0
	sq_radius = radius ** 2

	for x in range(center[1] - radius, center[1] + radius):
		for y in range(center[0] - radius, center[0] + radius):
			distance_from_center = (y- center[0])**2 + (x - center[1])**2
			if distance_from_center <= sq_radius:
				num_pixels_in_circle += 1

	return num_white_pixels / num_pixels_in_circle
			

def main(image_path, _search_length=1):
	global search_length
	search_length = _search_length

	originalImage = cv2.imread(image_path)
	bw_image = color_to_BW(originalImage) # convert color to BW image
	find_connected_components(bw_image)
	make_bounding_circle(originalImage)



if __name__ == "__main__":

	if len(sys.argv) != 3:
		print("Usage: python3 assn1.py <path_of_image> <search_length>")
		print("\nsearch_length : a positive integer")
		print("\t\thigher its value --> higher accuracy of center of bounding circle")
		print("\t\tat the same time, execution time also increases")
		print("\t\t5 is usually found to be good estimate with execution time less than 2 seconds")
		exit(1)

	image_path = sys.argv[1]
	_search_length = max(1, int(sys.argv[2]))
	main(image_path, _search_length)
