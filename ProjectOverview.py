import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from model import FacialExpressionModel
from model import GenderModel
import numpy as np
import tensorflow as tf

color = "rgb"
bins = 16
resizeWidth = 0
# Initialize plot.
fig, ax = plt.subplots()
if color == 'rgb':
    ax.set_title('Histogram (RGB)')
else:
    ax.set_title('Histogram (grayscale)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Frequency')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 6
alpha = 0.5
if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)
    lr, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#654321', lw=lw, alpha=alpha)
    lbr, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#123456', lw=lw, alpha=alpha)
    lg, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#225522', lw=lw, alpha=alpha)
else:
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
plt.ion()
#plt.show()
bi=None
gi=None
ri=None

    
#harcascade file to detect Face
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#loading emaotion Detection Model
model = FacialExpressionModel("emotions_model.json", "emotions_dataset.h5")
g_model=GenderModel("models/gender_model.json","models/gender_prediction.h5")
font = cv2.FONT_HERSHEY_DUPLEX

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
flag=0;
demoimg=storage=0;
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    numPixels = np.prod(image.shape[:2])
    demoimg= np.zeros(image.shape, dtype="uint8")
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic.process(image)

    
    gray_fr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.2, 5)

    for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            pred_g=g_model.predict_gender(roi[np.newaxis, :, :, np.newaxis])
            cv2.putText(demoimg, "Expresion: "+pred, (10, 20), font, 0.6, (255, 255, 110), 1)
            cv2.putText(demoimg, "Gender: "+pred_g, (10, 40), font, 0.6, (255, 255, 110), 1)
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        demoimg,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())

    
    cv2.imshow("image",image)
    cv2.imshow("imga2",demoimg)
    (b, g, r) = cv2.split(image)
    
    if bi==0 or gi==0 or ri==0:
        bi=b
        gi=g
        ri=r
        histogramRi = cv2.calcHist([ri], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramGi = cv2.calcHist([gi], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramBi = cv2.calcHist([bi], [0], None, [bins], [0, 255]) / (numPixels/4)
        lr.set_ydata(histogramRi)
        lg.set_ydata(histogramGi)
        lbr.set_ydata(histogramBi)
    else:
        histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramRi = cv2.calcHist([ri], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramGi = cv2.calcHist([gi], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramBi = cv2.calcHist([bi], [0], None, [bins], [0, 255]) / (numPixels/4)
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
        fig.canvas.draw()
        lr.set_ydata(histogramRi)
        lg.set_ydata(histogramGi)
        lbr.set_ydata(histogramBi)
    fig.canvas.draw()
    fig.savefig('plot.jpg', bbox_inches='tight', dpi=150)
    plotteddata=cv2.resize(cv2.imread("plot.jpg"),(image.shape[1],image.shape[0]))
    cv2.imshow("plot",plotteddata)
    if cv2.waitKey(1) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()

