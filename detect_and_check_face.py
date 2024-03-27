import ast

import cv2
import numpy as np
from PIL import Image
from imgbeddings import imgbeddings
from sklearn.metrics import pairwise_distances_argmin_min

haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

with open('embeddings.txt', 'r') as text_file:
    file_content = text_file.read()

embeddings = np.array(ast.literal_eval(file_content))
text_file.close()

while rval:
    rval, frame = vc.read()
    key = cv2.waitKey(50)
    if key == 27:
        break

    faces = haar_cascade.detectMultiScale(
        frame, scaleFactor=1.05, minNeighbors=1, minSize=(100, 100)
    )
    if len(faces) > 0:
        print('face detected')
        x, y, w, h = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
        cropped_img = frame[y: y + h, x: x + w]

        face_img = Image.fromarray(cropped_img.astype('uint8'))

        ibed = imgbeddings()
        embedding = np.array(ibed.to_embeddings(face_img)).reshape(1, -1)
        arg_min, distances = pairwise_distances_argmin_min(embedding, embeddings)
        if distances[0] <= 10:
            break
        else:
            print('person not in embeddings')
vc.release()

print('found individual!!')