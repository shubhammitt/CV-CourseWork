'''
1. extract frames from video
2. find background frame using one of the strategy mean/median/mode
3. subtract bg from each frame and take average over channels
4. apply otsu on each changed frames
'''

import cv2
import sys
import copy
import numpy as np
from statistics import mode
from otsu import *


number_of_frames = 0
FPS = 0
path_input_frames = './input/'
path_intermediate_frames = './intermediate/'
path_output_frames = './output/'
image_extension = '.jpg'
extracted_frames = []
intermediate_frames = []
foreground_frames = []


def extract_frames(video_name):
	global number_of_frames, FPS

	vidcap = cv2.VideoCapture(video_name)
	FPS = vidcap.get(cv2.CAP_PROP_FPS)
	success, image = vidcap.read()

	while success:
		extracted_frames.append(image)
		cv2.imwrite(path_input_frames + str(number_of_frames) + image_extension, image)     # save frame as JPEG file      
		success,image = vidcap.read()
		number_of_frames += 1


def get_background_frame(background_strategy):

	if background_strategy == 'mean':
		background_frame = np.mean(np.array(extracted_frames), axis=0)
	elif background_strategy == 'mode':
		background_frame = np.apply_along_axis(mode, 0, np.array(extracted_frames))
	else:
		background_frame = np.median(np.array(extracted_frames), axis=0)

	background_frame = background_frame.astype(np.uint8)

	# cv2.imshow('Background frame', background_frame)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	return background_frame


def subtract_background_from_frames(background_frame):
	background_frame = background_frame.astype(np.int64)

	for i, frame in enumerate(extracted_frames):
		current_frame = frame.astype(np.int64)

		intermediate_frame = np.subtract(frame, background_frame)
		intermediate_frame = np.absolute(intermediate_frame)
		intermediate_frame = np.mean(intermediate_frame, axis=2).astype(np.int64)
		intermediate_frames.append(intermediate_frame)

		cv2.imwrite(path_intermediate_frames + str(i) + image_extension, intermediate_frame)


def apply_otsu_on_each_frame(min_threshold):
	i = 0
	for original_frame, intermediate_frame in zip(extracted_frames, intermediate_frames):
		foreground_frame = otsu(original_frame, intermediate_frame, 2, min_threshold)
		cv2.imwrite(path_output_frames + str(i) + image_extension, foreground_frame)
		height, width, layers = foreground_frame.shape
		size = ( width, height)
		foreground_frames.append(foreground_frame)
		i += 1
	return size


def convert_frames_to_video(size):
	# convert frames to video
	out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*"MJPG"), FPS, size)
	for frame in foreground_frames:
		out.write(frame)
	out.release()
	cv2.destroyAllWindows()


def main(video_name, background_strategy, min_threshold):
	extract_frames(video_name)
	background_frame = get_background_frame(background_strategy)
	subtract_background_from_frames(copy.deepcopy(background_frame))
	size = apply_otsu_on_each_frame(min_threshold)
	convert_frames_to_video(size)



if __name__ == '__main__':

	if len(sys.argv) != 4:
		print("Usage: python3 code.py <path_of_video> <background_strategy> <min_threshold>\n")
		print("background_strategy = mean/median/mode\n")
		print("min_threshold should be <0 to use otsu\n")
		exit(0)

	video_name = sys.argv[1]
	background_strategy = sys.argv[2]
	min_threshold = int(sys.argv[3])

	if min_threshold < 0 :
		min_threshold = None

	main(video_name, background_strategy, min_threshold)
