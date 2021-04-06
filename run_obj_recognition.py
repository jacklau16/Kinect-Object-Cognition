###################################################################################################
# File       : capture_train_img.py
# Description: Prepare training images by capturing a number of images of each object class
#              from the provided video frames
# Usage      : python capture_train_img.py <Training video dataset path>
###################################################################################################
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import cv2
import time
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from FreenectPlaybackWrapper.PlaybackWrapper import FreenectPlaybackWrapper
import obj_recogniser

def main():

    showRGBImg = True
    showDepthImg = False
    trueLabelsFileName = 'Set2Labels.txt'
    allLabelsFileName = 'AllLabels.txt'

    # Load and train the object recogniser first
    recogniser = obj_recogniser.ObjRecogniser()
    recogniser.train()

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

    depthThreshold = recogniser.depthThreshold
    x_offset, y_offset = -40, 20

    # Load the true labels
    labelFile = open(trueLabelsFileName, 'r')
    trueClassLabels = labelFile.read().splitlines()

    #w, h = 0, 0
    recogResult = []

    startTime = time.time()

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):
        frameCount += 1
        if classCount == len(trueClassLabels):
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
            ret, thresholded = cv2.threshold(depth, depthThreshold, 255, cv2.THRESH_BINARY_INV)
            x, y, w, h = cv2.boundingRect(thresholded)
            if (h > 0) & (w > 0):
                cv2.putText(depth, trueClassLabels[classCount], (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    #            cv2.rectangle(depth, (x,y), (x+w, y+h), (0,0,255))
                if showDepthImg:
                    cv2.imshow("Depth", depth)
                lastObjCount = frameCount
                newClassStarted = True
                #predictedLabel = ''

                # Test Run Object Recognition
                if (h >= recogniser.imagetteGridSize) & (w >= recogniser.imagetteGridSize) & (rgb is not None):
                    max_score = -1
                    max_idx = -1
                    score = np.zeros(len(recogniser.objName_train))
                    t0 = time.time()

                    imgTest = rgb[y+y_offset:y+y_offset+h, x+x_offset:x+x_offset+w, :]

                    if recogniser.resizeImage:
                        imgTest = cv2.resize(imgTest, (200, 200), interpolation=cv2.INTER_AREA)

                    if recogniser.equaliseHistogram:
                        imgTest[:, :, 0] = cv2.equalizeHist(imgTest[:, :, 0])
                        imgTest[:, :, 1] = cv2.equalizeHist(imgTest[:, :, 1])
                        imgTest[:, :, 2] = cv2.equalizeHist(imgTest[:, :, 2])

                    # Call the object recognition routine of the detector
                    scores = recogniser.recognise_img(imgTest)

                    # Get the class index having maximum score
                    max_idx = scores.argmax()

                    t1 = time.time()

                    if max(scores) > 0:
                        # Calculate confidence level = maximum score / total scores
                        conf_level = scores[max_idx] / np.sum(scores)

                        # Assign class label only if the confidence level > threshold value
                        if conf_level > recogniser.confLevelThreshold:
                            predictedLabel = recogniser.objName_train[max_idx]
                            recogResult.append([frameCount, trueClassLabels[classCount], predictedLabel, max(scores), conf_level])
                            print("Recognition:", trueClassLabels[classCount], recogniser.objName_train[max_idx], scores[max_idx], t1 - t0)
                        else:
                            predictedLabel = "UNRECOGNISED"
                            recogResult.append([frameCount, trueClassLabels[classCount], predictedLabel, 0, 0])
                            print("Recognition:", trueClassLabels[classCount], "[UNRECOGNISED]", 0, t1 - t0)
                    else:
                        # All scores are zero -> object cannot be identified
                        conf_level = 0
                        predictedLabel = "UNRECOGNISED"
                        recogResult.append([frameCount, trueClassLabels[classCount], predictedLabel, 0, 0])
                        print("Recognition:", trueClassLabels[classCount], "[UNRECOGNISED]", 0, t1 - t0)

                    if showRGBImg:
                        cv2.putText(rgb, predictedLabel + " ({:.0f}%)".format(conf_level*100), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        cv2.imshow("RGB", rgb)

        if (frameCount - lastObjCount > interClassFrameThresVal) & newClassStarted:
            classCount += 1
            newClassStarted = False

        # Check for Keyboard input.
        key = cv2.waitKey(10)

        # Exit the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    endTime = time.time()
    print("Duration:", endTime-startTime)

    # Export the recognition result to 'result.csv'
    dfResult = pd.DataFrame(recogResult)
    dfResult.columns = ['Frame#', 'True Label', 'Predicted Label', 'Score', 'Conf. Level']
    dfResult.to_csv("result.csv")

    # Plot the confusion matrices
    plot_cm(dfResult, allLabelsFileName)

    return 0

# Function to calculate and plot the confusion matrices
def plot_cm(dfResult, allLabelsFileName):

    labelFile = open(allLabelsFileName, 'r')
    classList = labelFile.read().splitlines()
#    results = pd.read_csv('result.csv')
    cm = confusion_matrix(dfResult['True Label'], dfResult['Predicted Label'], labels=classList)
    cm_normalised = confusion_matrix(dfResult['True Label'], dfResult['Predicted Label'], labels=classList,
                                     normalize='true')
    print("Overall accuracy:", cm.diagonal().sum() / cm.sum())
    df_cm = pd.DataFrame(cm, index=classList, columns=classList)
    df_cm_normalised = pd.DataFrame(cm_normalised, index=classList, columns=classList)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    ax = sn.heatmap(df_cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    ax.figure.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion matrix of object recognition")
    plt.show()

    # Plotting the normalised confusion matrix
    plt.figure(figsize=(10, 8))
    ax = sn.heatmap(df_cm_normalised, annot=True, fmt='.2f', cmap=plt.cm.Blues)
    ax.figure.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Normalised confusion matrix of object recognition")
    plt.show()



if __name__ == "__main__":
    exit(main())
