from argparse import ArgumentParser

import numpy as np
import pandas as pd
import time
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from FreenectPlaybackWrapper.PlaybackWrapper import FreenectPlaybackWrapper

import cv2
import concurrent.futures
import objdetection2

def main():

    showRGBImg = True
    showDepthImg = False

    # load and train first
    detector = objdetection2.ObjDetection2()
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
    interClassFrameThresVal = 70

    thresVal = 75
    x_offset, y_offset = -40, 20
    labelFile = open('Set2Labels.txt', 'r')
    classLabels = labelFile.read().splitlines()
    w, h = 0, 0
    recogResult = []

    startTime = time.time()

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):
        frameCount += 1
        if classCount == len(classLabels):
            break
        # If we have an updated RGB image, then display
        if status.updated_rgb:
            #if (h > 0) & (w > 0):
                #cv2.rectangle(rgb, (x+x_offset, y+y_offset), (x+x_offset+w, y+y_offset+h), (0, 0, 255))
            #if showImg:
            #    cv2.imshow("RGB", rgb)
            #print(f"{frameCount}: RGB")
            a = 1
        # If we have an updated Depth image, then display
        if status.updated_depth:
            #lastObjCount = count
            #cv2.imshow("Depth", depth)
            #print(f"{frameCount}: DEPTH")
            ret, thresholded = cv2.threshold(depth, thresVal, 255, cv2.THRESH_BINARY_INV)
            x, y, w, h = cv2.boundingRect(thresholded)
            #print(x, y, w, h)
            if (h>0) & (w>0):
                cv2.putText(depth, classLabels[classCount], (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    #            cv2.rectangle(depth, (x,y), (x+w, y+h), (0,0,255))
                if showDepthImg:
                    cv2.imshow("Depth", depth)
                lastObjCount = frameCount
                newClassStarted = True
                predictedLabel = ''

                # Test Run Object Recognition

                if (h >= detector.imagette_grid_size) & (w >= detector.imagette_grid_size) & (rgb is not None):
                    max_score = -1
                    max_idx = -1
                    score = np.zeros(len(detector.objName_train))
                    t0 = time.time()

                    imgTest = rgb[y+y_offset:y+y_offset+h, x+x_offset:x+x_offset+w,:]

                    if detector.resizeImage:
                        imgTest = cv2.resize(imgTest, (200, 200), interpolation=cv2.INTER_AREA)

                    if detector.equaliseHistogram:
                        imgTest[:, :, 0] = cv2.equalizeHist(imgTest[:, :, 0])
                        imgTest[:, :, 1] = cv2.equalizeHist(imgTest[:, :, 1])
                        imgTest[:, :, 2] = cv2.equalizeHist(imgTest[:, :, 2])


                    scores = detector.recognise_rgb2(imgTest)
                    max_idx = scores.argmax()
                    t1 = time.time()
                    if max(scores) > 0:
                        conf_level = scores[max_idx] / np.sum(scores)
                        if conf_level > 0.2:
                            predictedLabel = detector.objName_train[max_idx]
                            #recogResult.append([frameCount, classCount, max_idx, max(scores), conf_level])
                            recogResult.append([frameCount, classLabels[classCount], predictedLabel, max(scores), conf_level])
                            print("Recognition:", classLabels[classCount], detector.objName_train[max_idx], scores[max_idx], t1 - t0)
                        else:
                            predictedLabel = "UNRECOGNISED"
                            recogResult.append([frameCount, classLabels[classCount], predictedLabel, 0, 0])
                            print("Recognition:", classLabels[classCount], "[UNRECOGNISED]", 0, t1 - t0)
                    else:
                        # not recognised
                        conf_level = 0
                        predictedLabel = "UNRECOGNISED"
                        recogResult.append([frameCount, classLabels[classCount], predictedLabel, 0, 0])
                        print("Recognition:", classLabels[classCount], "[UNRECOGNISED]", 0, t1 - t0)

                    if showRGBImg:
                        cv2.putText(rgb, predictedLabel + " ({:.0f}%)".format(conf_level*100), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.imshow("RGB", rgb)
                        #cv2.imshow("RGB2", imgTest)

        if (frameCount - lastObjCount > interClassFrameThresVal) & newClassStarted:
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
    dfResult.columns = ['Frame#', 'True Label', 'Predicted Label', 'Score', 'Conf. Level']
    dfResult.to_csv("result.csv")
    plot_cm(classLabels)
    return 0

def plot_cm(classLabels):
    labelFile = open('AllLabels.txt', 'r')
    classList = labelFile.read().splitlines()
    results = pd.read_csv('result.csv')
    cm = confusion_matrix(results['True Label'], results['Predicted Label'], labels=classList)
    cm_normalised = confusion_matrix(results['True Label'], results['Predicted Label'], labels=classList, normalize='true')
    print("Overall accuracy:",cm.diagonal().sum() / cm.sum())
    df_cm = pd.DataFrame(cm, index=classList, columns=classList)
    df_cm_normalised = pd.DataFrame(cm_normalised, index=classList, columns=classList)
    #np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8))
    ax = sn.heatmap(df_cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    ax.figure.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion matrix of object recognition")
    plt.show()
    plt.figure(figsize=(10, 8))
    ax = sn.heatmap(df_cm_normalised, annot=True, fmt='.2f', cmap=plt.cm.Blues)
    ax.figure.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Normalised confusion matrix of object recognition")
    plt.show()



if __name__ == "__main__":
    exit(main())
