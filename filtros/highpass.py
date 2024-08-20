import cv2 as cv
import numpy as np

img = cv.imread("CR_hurt.webp", cv.IMREAD_GRAYSCALE)

def convolucao(imagem, filtro, padding=0, desl=1): #(matriz da imagem, matriz do filtro, quantidade de contornos ao redor da imagem (0 = nenhum), deslocamento de linhas de colunas filtro a cada passo)
    
    
    filtro = np.flipud(np.fliplr(filtro)) #rotacionar o filtro, dado numérico, para convoluçao         
    
    filx = filtro.shape[0] #num de linhas do filtro
    fily = filtro.shape[1] #num de colunas do filtro
    imx = imagem.shape[0] #num de linhas da imagem
    imy = imagem.shape[1] #num de colunas da imagem
    
    #aplicaçao de padding   
    if padding !=0:
        imgpad = np.zeros((imx + padding*2, imy + padding*2)) #imagem de 0's com dimensões adicionais de acordo com o parametro padding dado
        imgpad[int(padding):int(-1*padding), int(padding):int(-1*padding)] = np.float16(imagem) #atribui os valores da imagem à região interna ao padding
    else:
        imgpad = np.float16(imagem)
        
   
    resultx = int(((imx - filx + 2*padding)/desl) + 1)  #cálculo das dimensões da imagem convolucionada, para filtro numérico
    resulty = int(((imy - fily + 2*padding)/desl) + 1)
    
    resultado = np.zeros((resultx, resulty))
    
    for x in range(imx):  #iterando o filtro pela imagem - passo de cima para baixo
        if x > imx - filx: #caso o filtro vá a passar pelo limite de linhas - chegou no canto inferior direito termina convolução
            break
        
        if x % desl == 0:  #confirma que cada passo seguirá de acordo com a quantidade de deslocamento dado
            for y in range(imy): #da esquerda pra direita
                if y > imy - fily: #caso o filtro vá a passar pelo limite de colunas
                    break
                
                
                if y % desl == 0: #confirma que cada passo seguirá de acordo com a quantidade de deslocamento dado
                        resultado[x,y] = (filtro * imgpad[x: x + filx, y: y + fily]).sum() #convoluçao aplicada
                
    
    return np.uint8(resultado)


cv.imshow('original',img)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('original',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

laplace = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) #filtro laplace

sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #filtro sobel horizontal, vertical basta transpor e vice-versa
sobel_v = np.transpose(sobel_h)

prewitt_h = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) #filtro prewitt horizontal, vertical basta transpor e vice-versa
prewitt_v = np.transpose(prewitt_h)

img_lap = convolucao(img, laplace)

cv.imshow('filtro laplace',img_lap)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro laplace',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

img_sobelh = convolucao(img, sobel_h)

img_sobelv = convolucao(img, sobel_v)

img_sobel = img_sobelh + img_sobelv

cv.imshow('filtro sobel horizontal',img_sobelh)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro sobel horizontal',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

cv.imshow('filtro sobel vertical',img_sobelv)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro sobel vertical',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

cv.imshow('filtro sobel somado',img_sobel)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro sobel somado',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

img_preh = convolucao(img, prewitt_h)

img_prev = convolucao(img, prewitt_v)

img_prewitt = img_preh + img_prev

cv.imshow('filtro prewitt horizontal',img_preh)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro prewitt horizontal',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

cv.imshow('filtro prewitt vertical',img_prev)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro prewitt vertical',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

cv.imshow('filtro prewitt somado',img_prewitt)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro prewitt somado',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada