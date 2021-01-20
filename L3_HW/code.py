import cv2
import sys
import copy
import numpy as np
from statistics import mode, mean


number_of_frames = 0
path_input_frames = './input/'
path_output_frames = './output/'
image_extension = '.jpg'
extracted_frames = []


def extract_frames(video_name):
	global number_of_frames

	vidcap = cv2.VideoCapture(video_name)
	success, image = vidcap.read()

	while success:
		extracted_frames.append(image)
		cv2.imwrite(path_input_frames + str(number_of_frames) + image_extension, image)     # save frame as JPEG file      
		success,image = vidcap.read()
		number_of_frames += 1


def median(List):
	return sorted(List)[len(List) // 2]


def get_background_frame(background_strategy):

	rows, cols, channels = extracted_frames[0].shape

	background_frame = np.zeros((rows , cols, channels), np.uint8)

	for row in range(rows):
		for col in range(cols):
			for channel in range(channels):
				pixels = []
				
				for frame in extracted_frames:
					pixels.append(frame[row][col][channel])

				new_pixel = 0

				if background_strategy == 'mean':
					new_pixel = mean(pixels)
				elif background_strategy == 'mode':
					new_pixel = mode(pixels)
				else:
					new_pixel = median(pixels)

				background_frame[row][col][channel] = int(new_pixel)
				

	cv2.imshow('background_frame', background_frame)
	cv2.waitKey()
	cv2.destroyAllWindows()
	return background_frame

# def get_background_frame1(background_strategy):

# 	if background_strategy == 'mean':
# 		background_frame = np.mean(np.array(extracted_frames), dtype=np.int64, axis=0)
# 	elif background_strategy == 'mode':
# 		background_frame = np.mode(np.array(extracted_frames), dtype=np.int64, axis=0)
# 	else:
# 		background_frame = np.median(np.array(extracted_frames), dtype=np.int64, axis=0)

# 	cv2.imshow('background_frame', background_frame)
# 	cv2.waitKey()
# 	cv2.destroyAllWindows()
# 	return background_frame


if __name__ == '__main__':

	if len(sys.argv) != 3:
		print("Usage: python3 code.py <path_of_video> <background_strategy>\n")
		print("background_strategy = mean/median/mode")
		exit(0)

	video_name = sys.argv[1]
	background_strategy = sys.argv[2]

	extract_frames(video_name)
	bg_frame = get_background_frame(background_strategy)
