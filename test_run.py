#!/usr/bin/env python

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import time

from FreenectPlaybackWrapper.PlaybackWrapper import FreenectPlaybackWrapper

import cv2
import concurrent.futures
import objdetection

def main():

    showImg = False
    # Recognition result: Frame#, true label, predicted label
    recogResult = [['Frame#', 'True Label', 'Predicted Label', 'Score']]

    # load and train first
    detector = objdetection.ObjDetection()
    #detector.test_run_rgb()
    detector.train_all()

    parser = ArgumentParser(description="OpenCV Demo for Kinect Coursework")
    parser.add_argument("videofolder", help="Folder containing Kinect video. Folder must contain INDEX.txt.",
                        default="ExampleVideo", nargs="?")
    #parser.add_argument("--no-realtime", action="store_true", default=False)
    parser.add_argument("--no-realtime", action="store_true", default=True)
    args = parser.parse_args()

    frameCount = 0
    classCount = 0
    lastObjCount = 0
    newClassStarted = False

    thresVal = 80
    x_offset = -40
    y_offset = 30

    labelFile = open('Set1Labels.txt', 'r')
    classLabels = labelFile.read().splitlines()
    w = 0
    h = 0

    startTime = time.time()

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):
        frameCount += 1
        # If we have an updated RGB image, then display
        if status.updated_rgb:
            #if (h > 0) & (w > 0):
                #cv2.rectangle(rgb, (x+x_offset, y+y_offset), (x+x_offset+w, y+y_offset+h), (0, 0, 255))
            if showImg:
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
                cv2.putText(depth, classLabels[classCount], (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    #            cv2.rectangle(depth, (x,y), (x+w, y+h), (0,0,255))
                if showImg:
                    cv2.imshow("Depth", depth)
                lastObjCount = frameCount
                newClassStarted = True

                # Test Run Object Recognition

                if (h >= detector.imagette_grid_size) & (w >= detector.imagette_grid_size) & (rgb is not None):
                    max_score = -1
                    max_idx = -1
                    score = np.zeros(len(detector.objName_train))
                    t0 = time.time()

                    imgTest = rgb[y+y_offset:y+y_offset+h, x+x_offset:x+x_offset+w,:]
                    scores = detector.recognise_rgb2(imgTest)
                    #print("Scores:", scores)
                    max_idx = scores.argmax()
                    conf_level = scores[max_idx] / np.sum(scores)
                    recogResult.append([frameCount, classCount, max_idx, max(scores)])

                    cv2.putText(rgb, classLabels[max_idx] + " ({:.0f}%)".format(conf_level*100), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.imshow("RGB", rgb)
                    # with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    #     futures = []
                    #     for i in range(len(detector.objName_train)):
                    #         #futures.append(executor.submit(detector.recognise, thresholded[y:y + h, x:x + w], i))
                    #         futures.append(executor.submit(detector.recognise_rgb, rgb[y+y_offset:y+y_offset+h, x+x_offset:x+x_offset+w,:], i))
                    #         if showImg:
                    #             cv2.imshow("rgb2", rgb[y+y_offset:y+y_offset+h, x+x_offset:x+x_offset+w,:])
                    #     scores = []
                    #     for future in concurrent.futures.as_completed(futures):
                    #         #print("Thread:", future.result()[1], future.result()[0])
                    #         score[future.result()[1]] = future.result()[0]
                    #     max_idx = score.argmax()
                    #     recogResult.append([frameCount, classCount, max_idx, max(score)])

                    t1 = time.time()
                    print("Recognition: ", detector.objName_train[max_idx], scores[max_idx], t1-t0)

        if (frameCount - lastObjCount > 80) & newClassStarted:
            classCount += 1
            newClassStarted = False

        #print(count, lastObjCount, newClassStarted)
        # Check for Keyboard input.
        key = cv2.waitKey(10)

        # Break out of the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    endTime = time.time()
    print("Duration:", endTime-startTime)
    dfResult = pd.DataFrame(recogResult)
    dfResult.to_csv("result.csv")
    return 0

#def detect_edges(imgInput):

if __name__ == "__main__":
    exit(main())
