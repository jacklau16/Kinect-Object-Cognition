###################################################################################################
# File       : capture_train_img.py
# Description: Prepare training images by capturing a number of images of each object class
#              from the provided video frames
# Usage      : python capture_train_img.py <Training video dataset path>
###################################################################################################
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

    interClassFrameThresVal = 70
    interImgGap = 30
    thresVal = 75
    x_offset = -40
    y_offset = 20

    outDir = "./train_img/"
    labelFile = open('Set1Labels.txt', 'r')
    classLabels = labelFile.read().splitlines()

    w = 0
    h = 0

    # Loop through the video frames using Freenect library
    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):
        frameCount += 1
        # RGB image is updated
        if status.updated_rgb:
            # Display the RGB image
            cv2.imshow("RGB", rgb)

            # Capture image once per <interImgGrap> frames
            if ((frameCount - lastTrainImgFrame) >= interImgGap) & (h > 0) & (w > 0):
                # Output file format [Class ID]-[Image Index].jpg
                outFileName = outDir + f"{classCount+1}-{trainImgIdx}.jpg"
                # Only save the ROI with using the object boundary detected in the last Depth frame
                outImg = rgb[y+y_offset:y+y_offset+h, x+x_offset:x+x_offset+w,:]
                print("Saving train image as", outFileName)
                cv2.imwrite(outFileName, outImg)
                trainImgIdx += 1
                lastTrainImgFrame = frameCount

        # If we have an updated Depth image, then use it to capture the region of interest (ROI)
        if status.updated_depth:
            # Use cv2.threshold() to find out the object boundary
            ret, thresholded = cv2.threshold(depth, thresVal, 255, cv2.THRESH_BINARY_INV)
            x, y, w, h = cv2.boundingRect(thresholded)

            # If object is detected, highlight in the Depth image
            if (h > 0) & (w > 0):
                cv2.putText(depth, classLabels[classCount], (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(depth, (x,y), (x+w, y+h), (0,0,255))
                cv2.imshow("Depth", depth)
                lastObjFrameCount = frameCount
                newClassStarted = True

        # Increment the class ID (i.e. next object class) if no object is detected for a long time, e.g. 70 frames
        if (frameCount - lastObjFrameCount > interClassFrameThresVal) & newClassStarted:
            classCount += 1
            newClassStarted = False
            trainImgIdx = 1

        # Check for Keyboard input.
        key = cv2.waitKey(10)

        # Exit the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    return 0

if __name__ == "__main__":
    exit(main())
