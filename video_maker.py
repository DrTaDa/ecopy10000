import cv2
import numpy as np
import glob
import pathlib

frames_per_part = 5000

for part in range(13, 100):

    print(f"Making video part {part}")

    video_name = f"part{part}.avi"
    start = part * frames_per_part
    end = (part + 1) * frames_per_part

    print("Reading frames ...")
    img_array = []
    for t in range(start, end):
        filename = f"./frames/{t}.jpg"
        if pathlib.Path(filename).is_file():
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

    if len(img_array):
        print("Writing video ...")
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    else:
        print("No frame for this part.")
