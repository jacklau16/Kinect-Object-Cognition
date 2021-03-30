#!/usr/bin/env python

from argparse import ArgumentParser
from FreenectPlaybackWrapper.PlaybackWrapper import FreenectPlaybackWrapper

import cv2

def main():
    parser = ArgumentParser(description="OpenCV Demo for Kinect Coursework")
    parser.add_argument("videofolder", help="Folder containing Kinect video. Folder must contain INDEX.txt.",
                        default="ExampleVideo", nargs="?")
    #parser.add_argument("--no-realtime", action="store_true", default=False)
    parser.add_argument("--no-realtime", action="store_true", default=True)
    args = parser.parse_args()

    count = 0
    classcount = 0
    lastObjCount = 0
    newClassStarted = False

    thresVal = 80
    x_offset = -40
    y_offset = 30

    labelFile = open('Set1Labels.txt', 'r')
    classLabels = labelFile.read().splitlines()
    w =0
    h =0

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):
        count += 1
        # If we have an updated RGB image, then display
        if status.updated_rgb:
            #if (h > 0) & (w > 0):
            #    cv2.rectangle(rgb, (x+x_offset, y+y_offset), (x+x_offset+w, y+y_offset+h), (0, 0, 255))
            #rgb[:,:,0] = 0
            #rgb[:,:,1] = 0
            print(rgb.shape)
            cv2.imshow("RGB", rgb)
            #print(f"{count}: RGB")
        # If we have an updated Depth image, then display
        if status.updated_depth:
            #lastObjCount = count
            #cv2.imshow("Depth", depth)
            #print(f"{count}: DEPTH")
            ret, thresholded = cv2.threshold(depth, thresVal, 255, cv2.THRESH_BINARY_INV)
            x, y, w, h = cv2.boundingRect(thresholded)
            #print(x, y, w, h)
            if (h>0) & (w>0):
                cv2.putText(depth, classLabels[classcount], (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    #            cv2.rectangle(depth, (x,y), (x+w, y+h), (0,0,255))
                cv2.imshow("Depth", depth)
                lastObjCount = count
                newClassStarted = True

        if (count - lastObjCount > 80) & newClassStarted:
            classcount += 1
            newClassStarted = False

        print(count, lastObjCount, newClassStarted)
        # Check for Keyboard input.
        key = cv2.waitKey(10)

        # Break out of the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    return 0

if __name__ == "__main__":
    exit(main())