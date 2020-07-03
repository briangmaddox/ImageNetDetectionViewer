# ImageNetDetectionViewer
Really simple OpenCV program to display YOLO detections from DarkNet-trained models (COCO and OpenImages)

This viewer combines the outputs from pre-trained YOLO models on both the COCO and OpenImages data sets, performs non-maxima suppression to remove duplicates, and then displays the results in an OpenCV window.

The base of the viewer came from the very excellent www.pyimagesearch.com website.

During testing, I found that the pre-trained COCO model outperformed OpenImages, so this program favors COCO over OpenImages when detecting the same object classes.
## Getting Started
You'll need Python3 and at least OpenCV 4.2 installed for this to work.  Note I have only tried it on Linux Mint 20 / Ubuntu 20.04 so far.
### Prerequisites
The easiest way would be to create a virtual environment and then run
```
pip install -r requirements.txt
```

Next, you will need to download the actual pre-trained neural network files from the https://pjreddie.com/darknet/ and place them in the **models** directory.

I have already included .cfg and .names files for the models in there.  For OpenImages, I converted everything to lowercase and then renamed the classes that overlap with COCO labels.  This way we only keep the COCO detections when there are duplicate object classes between the two networks.

Download the pre-trained COCO model from Darknet by running
```
wget -O coco.weights https://pjreddie.com/media/files/yolov3.weights
```
in the **models** directory.

Next download the pre-trained OpenImages model from Darknet by running
```
wget -O openimages.weights https://pjreddie.com/media/files/yolov3-openimages.weights
```
in the **models** directory also.

## Running
```
usage: OpenCVPhotoDetectionViewer.py [-h] [-l LOCALIMAGE] [-u URL] [-uly MODELS] [-c CONFIDENCE] [-n NMSTHRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -l LOCALIMAGE, --localimage LOCALIMAGE
                        Local path to input image
  -u URL, --url URL     URL to image
  -uly MODELS, --models MODELS
                        Base path to models directory
  -c CONFIDENCE, --confidence CONFIDENCE
                        minimum probability to filter weak detections
  -n NMSTHRESHOLD, --nmsthreshold NMSTHRESHOLD
                        NMS threshold
```

You should either provide a file on disk with `--localimage` or a URL with `--url`.  Only one or the other can be used.

## Acknowledgments
* Dr. Adrian Adrian Rosebrock and his website at www.pyimagesearch.com  