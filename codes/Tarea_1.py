# Procesamiento Digital de Imagenes.
# Tarea 1.
# Jeckson Jaimes. 12-10446.
# Andres Suarez. 12-10925.

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Leer imagen
img = cv2.imread('../img/Test2.png')

# Convertir a escala de grises
imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Laplacian Filter
imgFILT = cv2.Laplacian(imgGRAY,cv2.CV_8U)
# Gaussian Filter
imgFILT1 = cv2.GaussianBlur(imgGRAY, (3,3), 0)
# Bilateral Filter
imgFILT2 = cv2.bilateralFilter(imgGRAY,9,75,75)

# plt.figure(),
# plt.subplot(1,3,1),
# plt.imshow(imgFILT,cmap='Greys'),
# plt.subplot(1,3,2),
# plt.imshow(imgFILT1,cmap='Greys'),
# plt.subplot(1,3,3),
# plt.imshow(imgFILT2,cmap='Greys'),
# plt.show()

# ----------------------EROSION DILATACION

# Erosión
erosion = np.ones((2,1), np.uint8)
imgEr = cv2.erode(imgGRAY, erosion, iterations=1)
plt.subplot(2,4,1),
plt.imshow(imgEr,cmap='Greys')
plt.title('Erosionada')
# Dilatacion 
dilatacion = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
imgErDi = cv2.dilate(imgEr, dilatacion, iterations=1)
plt.subplot(2,4,2),
plt.imshow(imgErDi,cmap='Greys')
plt.title('Dilatada')

# ---------------------DILATACION EROSION

# Dilatacion 
dilatacion = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
imgDi = cv2.dilate(imgGRAY, dilatacion, iterations=1)
plt.subplot(2,4,5),
plt.imshow(imgDi,cmap='Greys')
plt.title('Dilatada')

imgDiEr = cv2.Laplacian(imgDi,cv2.CV_8U)
# Erosión
erosion = np.ones((1,3), np.uint8)
#imgDiEr = cv2.erode(imgDi, erosion, iterations=2)
#imgDiEr = imgDi
plt.subplot(2,4,6),
plt.imshow(imgDiEr,cmap='Greys')
plt.title('Erosionada')



# ESTO DE CLOSING ES PARA RELLENAR LAS LETRAS PERO NO ES NECESARIO
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# plt.subplot(1,5,5),
# plt.imshow(closing,cmap='Greys')
# plt.title('Closing')
# Closing

#  Se le aplica un treshold para obtener la imagen binaria 
# threshDiEr = cv2.threshold(imgDiEr,0,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
threshDiEr = cv2.adaptiveThreshold(imgDiEr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# threshDiEr = cv2.adaptiveThreshold(imgFILT, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
plt.subplot(2,4,7),
plt.imshow(threshDiEr,cmap='Greys')
plt.title('Threshold Di-Er')

#  Se le aplica un treshold para obtener la imagen binaria 
# threshErDi = cv2.threshold(imgErDi,0,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
threshErDi = cv2.adaptiveThreshold(imgErDi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
plt.subplot(2,4,3),
plt.imshow(threshErDi,cmap='Greys')
plt.title('Threshold Er-Di')


# thresh = cv2.Canny(thresh,100,200)
# sobelx = cv2.Sobel(thresh,cv2.CV_64F,1,0,ksize=5)  # x
# sobely = cv2.Sobel(thresh,cv2.CV_64F,0,1,ksize=5)  # y

im2, contours2, hierarchy2 = cv2.findContours(imgErDi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
im3, contours3, hierarchy3 = cv2.findContours(imgFILT,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

img1 = img
img2 = img

r = 0
i = 0
bandera = False
for i in range(len(contours2)):
    cnt=contours2[i]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),1)
    # Estoy intentando este if para que cuando detecte una i no pinte el otro pedazo y que el contador no se 
    # vuelva mierda pero no esta funcionando!
    # if x+w<19:
    #     if bandera==False:
    #         bandera = True 
    #         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    #         r = r + 1
    #     else: 
    #         bandera = False
    # else:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    #     r = r + 1

plt.subplot(2,4,4),
plt.imshow(img1,cmap='Greys')
plt.title('Contours')

r = 0
i = 0
bandera = False
for i in range(len(contours3)):
    cnt=contours3[i]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),1)

plt.subplot(2,4,8),
plt.imshow(img2,cmap='Greys')
plt.title('Contours')


print("El número de carácteres en la imagen es: " + str(len(contours2)))

print("El número de carácteres en la imagen es: " + str(len(contours3)))
plt.show()
