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

    frameCount = 0
    classCount = 0
    lastObjFrameCount = 0
    lastTrainImgFrame = 0
    trainImgIdx = 1
    newClassStarted = False
    #trainImgCaptured = False

    interClassFrameThresVal = 70
    interImgGap = 30
    thresVal = 75
    x_offset = -40
    y_offset = 20
    equaliseHistogram = True

    outDir = "./train_rgb_hist/"
    labelFile = open('Set1Labels.txt', 'r')
    classLabels = labelFile.read().splitlines()
    #trainFrames = [301, 800, 1351, 1751, 2371, 3001, 3501, 4200, 4800, 5400, 6101, 6401, 7001, 7451]

    w = 0
    h = 0

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):
        frameCount += 1
        #if classCount == len(classLabels):
        #    break
        # If we have an updated RGB image, then display
        if status.updated_rgb:
            #if (h > 0) & (w > 0):
            #    cv2.rectangle(rgb, (x+x_offset, y+y_offset), (x+x_offset+w, y+y_offset+h), (0, 0, 255))
            cv2.imshow("RGB", rgb)
            print(f"Frame # {frameCount}: RGB")

            if ((frameCount - lastTrainImgFrame) >= interImgGap) & (h > 0) & (w > 0):
                # Output file format [Class ID]-[Image Index].jpg
                outFileName = outDir + f"{classCount+1}-{trainImgIdx}.jpg"
                # Only save the ROI with using the object boundary detected in the last Depth frame
                outImg = rgb[y+y_offset:y+y_offset+h, x+x_offset:x+x_offset+w,:]

                # if equaliseHistogram:
                #     outImg[:, :, 0] = cv2.equalizeHist(outImg[:, :, 0])
                #     outImg[:, :, 1] = cv2.equalizeHist(outImg[:, :, 1])
                #     outImg[:, :, 2] = cv2.equalizeHist(outImg[:, :, 2])
                print("Saving train image as", outFileName)
                cv2.imwrite(outFileName, outImg)
                trainImgIdx += 1
                lastTrainImgFrame = frameCount

        # If we have an updated Depth image, then use it to capture the region of interest
        if status.updated_depth:
            # Use cv2.threshold() to find out the object boundary
            ret, thresholded = cv2.threshold(depth, thresVal, 255, cv2.THRESH_BINARY_INV)
            x, y, w, h = cv2.boundingRect(thresholded)


            if (h > 0) & (w > 0):
                cv2.putText(depth, classLabels[classCount], (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                cv2.rectangle(depth, (x,y), (x+w, y+h), (0,0,255))
                cv2.imshow("Depth", depth)
                lastObjFrameCount = frameCount
                newClassStarted = True

        if (frameCount - lastObjFrameCount > interClassFrameThresVal) & newClassStarted:
            classCount += 1
            newClassStarted = False
            trainImgIdx = 1

        #print(count, lastObjFrameCount, newClassStarted)
        # Check for Keyboard input.
        key = cv2.waitKey(10)

        # Break out of the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    return 0

if __name__ == "__main__":
    exit(main())
