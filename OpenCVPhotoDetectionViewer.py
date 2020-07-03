#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
Program to show object detections in images.

This program will download a file from the command-line url, perform object detection on it, and display it to the user
using the default OpenCV viewer.  Many thanks to Adrian at PyImageSearch.com for his many tutorials and contributions!

This is based on some of his examples and demonstrates how to combine the outputs from two different DarkNet-trained
models.
"""

import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np


# **********************************************************************************************************************
def LoadImagefromURL(inURL: str, inFlags):
    """
    Load an image from the passed-in URL, convert to a numpy array, and decode as an OpenCV object
    :param inURL: URL for the image
    :param inFlags: Flags to pass to OpenCV
    :return:
    """

    # Create our urllib object
    req = urllib.request.Request(inURL)

    # Create the buffer
    buffer = urllib.request.urlopen(req)

    # Convert the buffer into a numpy array
    npArray = np.frombuffer(buffer.read(), dtype=np.uint8)

    # Return the image
    return cv2.imdecode(npArray, inFlags)


# **********************************************************************************************************************
def main():
    """ Main function of the program.
    """
    try:
        # Construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-l", "--localimage", type=str, default="", help="Local path to input image")
        ap.add_argument("-u", "--url", type=str, default="", help="URL to image")
        ap.add_argument("-uly", "--models", default="./models", help="Base path to models directory")
        ap.add_argument("-c", "--confidence", type=float, default=0.10,
                        help="minimum probability to filter weak detections")
        ap.add_argument("-n", "--nmsthreshold", type=float, default=0.30, help="NMS threshold")
        args = vars(ap.parse_args())

        # Check if we have any inputs
        localImage = args["localimage"]
        url = args["url"]

        # Check if both inputs are empty
        if not localImage and not url:
            print("Error!  You must specify either an input image or an input url.  Exiting...")
            sys.exit(-1)

        # Check if we specified both input types
        if localImage and url:
            print("Error!  You must only specify an input local image or an input url, not both.  Exiting...")
            sys.exit(-1)

        THRESHOLD = args["nmsthreshold"]  # NMS threshold
        CONFIDENCE = args["confidence"]  # Minimum detection confidence
        objectList = list()  # List to hold the actual object categories
        imageMat = None

        # Load the COCO class labels our YOLO model was trained on
        cocoLabelsPath = os.path.sep.join([args["models"], "coco.names"])
        openImagesLabelsPath = os.path.sep.join([args["models"], "openimages.names.lowercase"])
        COCOLABELS = open(cocoLabelsPath).read().strip().split("\n")
        OPENIMAGESLABELS = open(openImagesLabelsPath).read().strip().split("\n")

        # Initialize a list of colors to represent each possible class label.  Note that we have a total of 607
        # unique objects by combining both COCO and OpenImages
        np.random.seed(8675309)
        COLORS = np.random.randint(0, 255, size=(607, 3), dtype="uint8")

        # Read in our weights and cfg files
        cocoWeightsPath = os.path.sep.join([args["models"], "coco.weights"])
        cocoConfigPath = os.path.sep.join([args["models"], "coco.cfg"])
        openImagesWeightsPath = os.path.sep.join([args["models"], "openimages.weights"])
        openImagesConfigPath = os.path.sep.join([args["models"], "openimages.cfg"])

        # Load our YOLO object detector trained on COCO dataset (80 classes)
        print("Loading COCO from disk")
        cocoNet = cv2.dnn.readNetFromDarknet(cocoConfigPath, cocoWeightsPath)
        # Load our YOLO object detector trained on OpenImages dataset (601 classes)
        print("Loading OpenImages from disk")
        openImagesNet = cv2.dnn.readNetFromDarknet(openImagesConfigPath, openImagesWeightsPath)

        # Get the image matrix from the file
        try:
            if localImage:
                imageMat = cv2.imread(localImage, cv2.IMREAD_COLOR)
            else:
                imageMat = LoadImagefromURL(url, cv2.IMREAD_COLOR)
        except Exception as readExcept:
            print("Exception occurred while reading the image.  Exception is: {}".format(readExcept))
            sys.exit(-1)

        # Get the width and height of the image
        (imageHeight, imageWidth) = imageMat.shape[:2]

        # Get the COCO and OpenImages output layer names
        cocoLayerNames = cocoNet.getLayerNames()
        cocoLayerNames = [cocoLayerNames[i[0] - 1] for i in cocoNet.getUnconnectedOutLayers()]
        openImagesLayerNames = openImagesNet.getLayerNames()
        openImagesLayerNames = [openImagesLayerNames[i[0] - 1] for i in openImagesNet.getUnconnectedOutLayers()]

        # Construct a blob from the input image and then perform a forward pass of the YOLO object detector
        print("Performing forward detections...")
        blob = cv2.dnn.blobFromImage(imageMat, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        cocoNet.setInput(blob)
        openImagesNet.setInput(blob)
        cocoLayerOutputs = cocoNet.forward(cocoLayerNames)
        openImagesLayerOutputs = openImagesNet.forward(openImagesLayerNames)
        print("Forward detections done.")

        # Initialize our lists of detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        classIDs = []

        # Loop over each of the layer outputs in coco
        for output in cocoLayerOutputs:
            # Loop over each of the detections
            for detection in output:
                # Extract the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Only include if it is >= our confidence level
                if confidence >= args["confidence"]:
                    # Get the coordinates of the bounding box and then get the centers coordinates and width, height
                    box = detection[0:4] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                    (centerX, centerY, boxWidth, boxHeight) = box.astype("int")

                    # Get the upper left ulx and uly coordinates of the bounding box.
                    ulx = int(centerX - (boxWidth / 2))
                    uly = int(centerY - (boxHeight / 2))

                    # Update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([ulx, uly, int(boxWidth), int(boxHeight)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    objectList.append(COCOLABELS[classID])

        # Loop over each of the layer outputs in openimages
        for output in openImagesLayerOutputs:
            # Loop over each of the detections
            for detection in output:
                # Extract the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # If the object label is in COCO, skip this one as we prefer the overlapping detections from COCO
                if OPENIMAGESLABELS[classID] in COCOLABELS:
                    continue

                # Only include if it is >= our confidence level
                if confidence > args["confidence"]:
                    # Get the coordinates of the bounding box and then get the centers coordinates and width, height
                    box = detection[0:4] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                    (centerX, centerY, boxWidth, boxHeight) = box.astype("int")

                    # Get the upper left ulx and uly coordinates of the bounding box.
                    ulx = int(centerX - (boxWidth / 2))
                    uly = int(centerY - (boxHeight / 2))

                    # Update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([ulx, uly, int(boxWidth), int(boxHeight)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    objectList.append(OPENIMAGESLABELS[classID])

        # Apply non-maxima suppression to our indices to remove overlapping/duplicate results.
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        # Ensure at least one detection exists
        if len(idxs) > 0:
            # Loop over the indexes we are keeping
            for i in idxs.flatten():
                # Extract the bounding box coordinates
                (ulx, uly) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(imageMat, (ulx, uly), (ulx + w, uly + h), color, 2)
                text = "{}".format(objectList[i])
                cv2.putText(imageMat, text, (ulx, uly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the output image
        cv2.imshow("Image", imageMat)
        cv2.waitKey(0)

    except Exception as mainExcept:
        print("Exception occurred while running.  Exception is: {}".format(mainExcept))
        sys.exit(-1)


# **********************************************************************************************************************
if __name__ == "__main__":
    main()
