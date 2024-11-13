from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

main = tkinter.Tk()
main.title("Facial Expression from Webcam")
main.geometry("1200x1200")

emotion =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
exp_model = keras.models.load_model("model/model_35_91_61.h5")
font_cv = cv2.FONT_HERSHEY_SIMPLEX
face_cas = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
global video
names = []
ids = []
recognizer = cv2.face_LBPHFaceRecognizer.create()

def getImagesAndLabels():
    global names, ids
    names.clear()
    ids.clear()
    faces = []
    imagePaths = [os.path.join("person_details",f) for f in os.listdir("person_details")]
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage,'uint8')
        fname = os.path.basename(imagePath)
        arr = fname.split("_")
        Id = int(arr[1])
        names.append(arr[0])
        faces.append(imageNp)
        ids.append(Id)        
    return faces,ids,names

def getName(Id,names,ids):
    name = "Unable to predict name"
    for i in range(len(ids)):
      if ids[i] == Id:
        name = names[i]
        break
    return name
  
def facialExpression():
    global video
    faces,ids,names = getImagesAndLabels()
    recognizer.read("model/Trainner.yml")
    video = cv2.VideoCapture(0)
    while(True):
        ret, frame = video.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cas.detectMultiScale(gray, 1.3,5)
            for (x, y, w, h) in faces:
                face_component = gray[y:y+h, x:x+w]
                fc = cv2.resize(face_component, (48, 48))
                inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
                inp = inp/255.
                prediction = exp_model.predict_proba(inp)
                em = emotion[np.argmax(prediction)]
                score = np.max(prediction)
                Id, conf = recognizer.predict(face_component)
                if(conf < 50):
                    cv2.putText(frame, "Name: "+getName(Id,names,ids), (x, y-50), font_cv, 1, (0, 255, 0), 2)
                cv2.putText(frame, em+"  "+str(score*100)+'%', (x, y), font_cv, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imshow("image", frame)
            if cv2.waitKey(250) & 0xFF == ord('q'):
                break    
        else:
            break
    video.release()
    cv2.destroyAllWindows()


                

font = ('times', 14, 'bold')
title = Label(main, text='Facial Expression from Webcam')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
'''
l1 = Label(main, text='PERSON ID')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=30)
tf1.config(font=font1)
tf1.place(x=150,y=100)

l2 = Label(main, text='PERSON NAME')
l2.config(font=font1)
l2.place(x=50,y=150)

tf2 = Entry(main,width=60)
tf2.config(font=font1)
tf2.place(x=150,y=150)


imageButton = Button(main, text="Capture Persons & Train Model", command=trainPerson)
imageButton.place(x=50,y=200)
imageButton.config(font=font1)  
'''
pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=480,y=200)

videoButton = Button(main, text="Person & Facial Expression Recognition", command=facialExpression)
videoButton.place(x=50,y=250)
videoButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=400)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
