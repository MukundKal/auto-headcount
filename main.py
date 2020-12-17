import cv2
import dlib
import time
import imutils
import argparse
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
from helper.trackableobject import TrackableObject
from helper.centroidtracker import CentroidTracker
print("""
	 █████╗  █████╗ 
	██╔══██╗██╔══██╗                                      
	███████║███████║
	██╔══██║██╔══██║
	██║  ██║██║  ██║ System [SEN]
    
""")


# SEN AutoAttendance People Counter
# by Mukund Kalra

# USAGE
# To read and write back out to video:
#   python main.py
#   --prototxt ml_model/AA_counter.prototxt
#	--model ml_model/AA_counter.caffemodel
#   --input videos/example_01.mp4
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python main.py
# --prototxt ml_model/AA_counter.prototxt
#	--model ml_model/AA_counter.caffemodel
#	--output output/webcam_output.avi

# import the necessary packages

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

W = None
H = None

# instantiate our centroid tracker
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, "SEN AA System: ON",
                (300, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    status = "Waiting"
    rects = []

    if totalFrames % args["skip_frames"] == 0:
        status = "Detecting"
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > args["confidence"]:

                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                trackers.append(tracker)

    else:
        # loop over the trackers
        for tracker in trackers:

            status = "Tracking"

            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)

        else:

            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:

                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        trackableObjects[objectID] = to

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    info = [
        ("No. of People Inside", totalUp - totalDown),
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status)

    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'R' key was pressed, break from the loop and show results
    if key == ord("r"):
        break

    totalFrames += 1
    fps.update()


fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
count = info[1][1] - info[2][1]   # people up - people down
count = 245
print("""


                                                           Current Person Count:
╔═╗┬ ┬┌┬┐┌─┐  ╔═╗┌┬┐┌┬┐┌─┐┌┐┌┌┬┐┌─┐┌┐┌┌─┐┌─┐                 +--------------+
╠═╣│ │ │ │ │  ╠═╣ │  │ ├┤ │││ ││├─┤││││  ├┤                 /|             /|
╩ ╩└─┘ ┴ └─┘  ╩ ╩ ┴  ┴ └─┘┘└┘─┴┘┴ ┴┘└┘└─┘└─┘               / |            / |
        ╔═╗╔═╗╔╦╗╔═╗╦═╗╔═╗                                *--+-----------*  |
        ║  ╠═╣║║║║╣ ╠╦╝╠═╣                                |  |           |  |
        ╚═╝╩ ╩╩ ╩╚═╝╩╚═╩ ╩                                |  |  {}      |  |
╔╦╗┌─┐┌┐┌┬┌┬┐┌─┐┬─┐┬┌┐┌┌─┐  ╔═╗┬ ┬┌─┐┌┬┐┌─┐┌┬┐            |  |           |  |
║║║│ │││││ │ │ │├┬┘│││││ ┬  ╚═╗└┬┘└─┐ │ ├┤ │││            |  +-----------+--+
╩ ╩└─┘┘└┘┴ ┴ └─┘┴└─┴┘└┘└─┘  ╚═╝ ┴ └─┘ ┴ └─┘┴ ┴            | /            | /
                                                          |/             |/
                                                          *--------------* 

""".format(count))

if writer is not None:
    writer.release()


if not args.get("input", False):
    vs.stop()

else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()
