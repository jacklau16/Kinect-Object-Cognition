#!/usr/bin/env python

from argparse import ArgumentParser
from FreenectPlaybackWrapper.PlaybackWrapper import FreenectPlaybackWrapper

import cv2


def main():
    parser = ArgumentParser(description="OpenCV Demo for Kinect Coursework")
    parser.add_argument("videofolder", help="Folder containing Kinect video. Folder must contain INDEX.txt.",
                        default="ExampleVideo", nargs="?")
    parser.add_argument("--no-realtime", action="store_true", default=False)

    args = parser.parse_args()

    for status, rgb, depth in FreenectPlaybackWrapper(args.videofolder, not args.no_realtime):

        # If we have an updated RGB image, then display
        if status.updated_rgb:
            cv2.imshow("RGB", rgb)

        # If we have an updated Depth image, then display
        if status.updated_depth:
            cv2.imshow("Depth", depth)

        # Check for Keyboard input.
        key = cv2.waitKey(10)

        # Break out of the program if ESC is pressed (OpenCV KeyCode 27 is ESC key)
        if key == 27:
            break

    return 0

if __name__ == "__main__":
    exit(main())
