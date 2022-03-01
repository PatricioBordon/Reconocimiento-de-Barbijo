import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import tensorflow as tf
from scipy.special import softmax



modelo_deteccion_cara = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt','./models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
#Modelo de deteccion de barbijo
Modelo = tf.keras.models.load_model('Modelo_cara_CNN/')

#Informacion de las etiquetas
etiquetas = ['Barbijo bien puesto', 'Sin Barbijo', 'Falta cubrir nariz', 'Falta cubrir menton']

def getColor (etiqueta):
    if etiqueta =='Barbijo bien puesto':
        color=(0,255,0)

    elif etiqueta == 'Sin Barbijo':
        color = (0,0,255)

    elif etiqueta == '':
        color = (0,0,0)

    else:
        color = (0,0,0)
    return color

#Reconocimiento
#.1_Deteccion de cara
def prediccion (imagen):
    
    img = imagen.copy()
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,117,123),swapRB=True)
    modelo_deteccion_cara.setInput(blob)
    deteccion = modelo_deteccion_cara.forward()
    for i in range(0,deteccion.shape[2]):
        confiabilidad = deteccion[0,0,i,2]
        if confiabilidad > 0.5:
            caja = deteccion[0,0,i,3:7]*np.array([w,h,w,h])
            caja = caja.astype(int)
            pt1 = (caja[0], caja[1])
            pt2 = (caja[2], caja[3])
            #cv2.rectangle(imagen,pt1,pt2,(0,255,0),2)
            #.2_Preprocesar datos
            cara = imagen[caja[1]:caja[3],caja[0]:caja[2]]
            cara_blob = cv2.dnn.blobFromImage(cara,1,(100,100),(104,117,123),swapRB=True)
            cara_blob_squeeze = np.squeeze(cara_blob).T
            cara_blob_rotar = cv2.rotate(cara_blob_squeeze,cv2.ROTATE_90_CLOCKWISE)
            cara_blob_voltear = cv2.flip(cara_blob_rotar,1)
            #Normalizacion
            img_norm = np.maximum(cara_blob_voltear,0)/cara_blob_voltear.max()
            #.3_Aprendizaje Profundo
            img_Input = img_norm.reshape(1,100,100,3)
            Resultado = Modelo.predict(img_Input)
            Resultado = softmax(Resultado)[0]
            Confianza_Indice = Resultado.argmax()
            Confianza_Puntaje = Resultado[Confianza_Indice]
            etiqueta = etiquetas[Confianza_Indice]
            etiqueta_texto = '{}'.format(etiqueta,)
            #print(etiqueta_texto)
            #Mostrar cuadro
            color = getColor(etiqueta)
            cv2.rectangle(imagen, pt1, pt2, color, 1)
            cv2.putText(imagen, etiqueta_texto, pt1, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    return imagen
        