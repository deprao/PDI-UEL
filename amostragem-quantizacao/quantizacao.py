import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('abraxas.jpg')

cv.imshow('quantizacao',img)         #amostra de imagem inicial
cv.moveWindow('quantizacao',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)

img_gray = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint8) #converter para grayscale pela média de cada pixel, limitando a um máximo de 256 tons de cinza
for i in np.arange(img.shape[0]):
        for j in np.arange(img.shape[1]):
            img_gray[i][j] = np.mean(img[i][j])

sample_original = 2**8        #intensidade original
sample_factor = sample_original #fator de amostra - apenas para indicar a fase de quantização no título da imagem

scale_factor = 1

while True :
    factor_bits = int(np.log2(sample_factor))
    cv.imshow('quantizacao - grayscale nbits = '+str(factor_bits),img_gray)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
    cv.moveWindow('quantizacao - grayscale nbits = '+str(factor_bits),960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
    cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

    scale_factor *= 2

    img_gray = np.uint8(img_gray/scale_factor) * scale_factor
    sample_factor /= 2
    
    if sample_original == scale_factor :
        break

