import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('CR_hurt.webp')

img_gray = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint8) #converter para grayscale pela média de cada pixel, limitando a um máximo de 256 tons de cinza
for i in np.arange(img.shape[0]):
        for j in np.arange(img.shape[1]):
            img_gray[i][j] = np.mean(img[i][j])

cv.imshow('original',img_gray)         #amostra de imagem inicial
cv.moveWindow('original',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)

hist, bins = np.histogram(img_gray.flatten(),256,[0,256]) #histograma da imagem, retornando a quantidade de valores distribuídos pelo eixo x (bins; valores de cores)

cdf = hist.cumsum() #função de distribuição acumulativa
cdf_norm = (cdf * float(hist.max()) / cdf.max()) #normaliza a cdf

plt.subplot(221),plt.hist(img_gray.flatten(),256,[0,256],density=True),plt.title('histograma da imagem normalizado') #plot do histograma original normalizado

plt.subplot(222),plt.plot(cdf_norm),plt.title('função de distribuição acumulativa') #plot da cdf normalizada

cdf_mask = np.ma.masked_equal(cdf,0) #cria uma máscara da cdf para ignorar valores zerados, a fim de encontrar a frequência mínima para aplicar equalização

cdf_mask = (cdf_mask - cdf_mask.min())*255/(cdf_mask.max()-cdf_mask.min())  #transformação da cdf

cdf = np.ma.filled(cdf_mask,0).astype('uint8') #preenche as máscaras de volta com zeros

img_eq = cdf[img_gray] #mapeamento da transformação para equalizar imagem

hist_eq, bins = np.histogram(img_eq.flatten(),256,[0,256]) #histograma da imagem equalizada, retornando a quantidade de valores distribuídos pelo eixo x (bins; valores de cores)

cdf = hist_eq.cumsum() #função de distribuição acumulativa do histograma equalizado
cdf_norm = (cdf * float(hist_eq.max()) / cdf.max()) #normaliza a cdf equalizada

plt.subplot(223),plt.hist(img_eq.flatten(),256,[0,256],density=True),plt.title('histograma equalizado, normalizado') #plot do histograma equalizado, normalizado

plt.subplot(224),plt.plot(cdf_norm),plt.title('função de distribuição acumulativa após equalização') #plot da cdf da equalização normalizada

plt.show()

cv.imshow('equalizada',img_eq)         #amostra de imagem equalizada
cv.moveWindow('equalizada',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)
