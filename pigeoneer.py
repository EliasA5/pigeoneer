import cv2 as cv
import json
import numpy
import argparse
import datetime
import time
import warnings
import sys
from pygame import mixer



ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required= True, help = "Path to Config file")
args = vars(ap.parse_args())


conf = json.load(open(args["conf"]))

vs = cv.VideoCapture(0)

mixer.init()
mixer.music.load('hello_there.mp3')
sound_ts = datetime.datetime.now()

if not vs.isOpened():
    print("Cant Open Camera")
    sys.exit()

vs.set(3, conf["resolution"][0])
vs.set(4, conf["resolution"][1])
vs.set(5, 16)


print("Camera Warming Up")
time.sleep(conf["camera_warmup_time"])
avg = None
i = 0
while vs.isOpened():
    ret, frame = vs.read()

    if i != conf["fps"]:
        i = i+1
        continue

    i = 0
    detected = False
    if(ret == False or cv.waitKey(1) & 0xFF == ord('q')):
        break

    timestamp = datetime.datetime.now()
    pigeon = "no Pigeon detected"

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    if avg is None:
        avg = gray.copy().astype("float")

    cv.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv.absdiff(gray, cv.convertScaleAbs(avg))
    thresh = cv.threshold(frameDelta, conf["delta_thresh"], 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations = 2)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    for cont in cnts:
        if cv.contourArea(cont) < conf["min_area"]:
            continue
        (x, y, w, h) = cv.boundingRect(cont)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pigeon = "Pigeon detected"
        detected = True

    
        
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv.putText(frame, "Pigeon Status: {}".format(pigeon), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv.putText(frame, ts, (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    if conf["play_sound"] and detected and (abs(datetime.datetime.now() - sound_ts) > datetime.datetime.timedelta(minutes = 1)):
        sound_ts = datetime.datetime.now()
        mixer.music.play()

    if conf["show_video"]:
        cv.imshow("Pigeon Feed", frame)
        cv.imshow("gray", gray)
        cv.imshow("thresh_hold", thresh)

print("Shutting Down")
vs.release()

cv.destroyAllWindows()


        

