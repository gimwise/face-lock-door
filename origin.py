mport face_recognition
import cv2
import numpy as np

from picamera import PiCamera
from time import sleep
import datetime
import sys, os
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from uuid import uuid4
import RPi.GPIO as GPIO
import time

PROJECT_ID = "example-fedcc"

cred = credentials.Certificate("/home/pi/Series3_Camera(FireStorage)/serviceAccountKey.json")
default_app = firebase_admin.initialize_app(cred, {
    'storageBucket': f"{PROJECT_ID}.appspot.com"
})


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)


obama_image = face_recognition.load_image_file("pic.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


biden_image = face_recognition.load_image_file("an.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]


known_face_encodings = [
    obama_face_encoding,
    #biden_face_encoding
]

known_face_names = [
    "Won-dam",
    #"yu-jin"
]

prev_name = "NULL"
name = "NULL"

def face():
    while True:
        global face_locations
        global face_encodings 
        global face_names 
        global process_this_frame
        global video_capture 
        global obama_image 
        global obama_face_encoding 
        global biden_image 
        global biden_face_encoding 
        global known_face_encodings
        global known_face_names

        global prev_name
        global name
        
        # Grab a single frame of video

        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

     

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

        rgb_small_frame = small_frame[:, :, ::-1]

     

        # Only process every other frame of video to save time

        if process_this_frame:

            # Find all the faces and face encodings in the current frame of video

            face_locations = face_recognition.face_locations(rgb_small_frame)

            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

     

            face_names = []

            for face_encoding in face_encodings:

                # See if the face is a match for the known face(s)

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                

     

                # # If a match was found in known_face_encodings, just use the first one.

                # if True in matches:

                #     first_match_index = matches.index(True)

                #     name = known_face_names[first_match_index]

     

                # Or instead, use the known face with the smallest distance to the new face

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    
                    prev_name = name

                    name = known_face_names[best_match_index]

                else:
                    
                    prev_name = name
                    
                    name = "Unknown"
                    

                face_names.append(name)

     

        process_this_frame = not process_this_frame

     

     

        # Display the results

        for (top, right, bottom, left), name in zip(face_locations, face_names):

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size

            top *= 4

            right *= 4

            bottom *= 4

            left *= 4

     

            # Draw a box around the face

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

     

            # Draw a label with a name below the face

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            #excute_camera()

            #open door
            
            

            

     

        # Display the resulting image

        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
           
        controlDoor(prev_name, name)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

     

    # Release handle to the webcam

    video_capture.release()

    cv2.destroyAllWindows()
    

def controlDoor(prev_name, name):

    doorOpen()
    
  #  if prev_name != name and prev_name == 'NULL' and prev_name == 'Unknown' and name != 'NULL' and name != 'Unknown':
   #     doorOpen()
   #     print("%s %s OPEN DOOR"%(prev_name, name))
   # else:
    #    print("%s %s"%(prev_name, name))
    
    
   

    
def doorOpen():
    # Servo Motor
    
    servo_pin = 18
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servo_pin,GPIO.OUT)
    pwm = GPIO.PWM(servo_pin, 50)
    pwm.start(5)
    
    for high_time in range(50, 110) :
        pwm.ChangeDutyCycle(high_time/10.0)
        time.sleep(0.01)


    for high_time in range(110, 50, -1) :
        pwm.ChangeDutyCycle(high_time/10.0)
        time.sleep(0.01)
    
    GPIO.cleanup()
    pwm.stop()


def fileUpload(file):
    bucket = storage.bucket()
    blob = bucket.blob('Images/'+file)
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}
    blob.metadata = metadata

    blob.upload_from_filename(filename='/home/pi/Images/'+file, content_type='image/jpg')
    print(blob.public_url)
    
'''
def execute_camera():
    
    subtitle = "Ras"
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.jpg'
    filename = "_".join([subtitle, suffix])
    
    cv2.capture('/home/pi/Images/' + filename)
    print("Server Uploading...")
    fileUpload(filename)
    print("Done!")
'''


def main():
    
    face()
        
    # Release handle to the webcam
    #video_capture.release()
    #cv2.destroyAllWindows()
    
    
if __name__== '__main__':
    main()