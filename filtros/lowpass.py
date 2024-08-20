import cv2 as cv
import numpy as np
from scipy import stats as st

img = cv.imread("CR_hurt.webp", cv.IMREAD_GRAYSCALE)

def convolucao(imagem, filtro, padding=0, desl=1, lim=None): #(matriz da imagem, matriz do filtro, quantidade de contornos ao redor da imagem (0 = nenhum), deslocamento de linhas de colunas filtro a cada passo, limiar para a média caso fornecido)
    
    tipo_filtro = 'media'
    
    if (filtro != 'moda') & (filtro != 'mediana'):
        filtro = np.flipud(np.fliplr(filtro)) #rotacionar o filtro, dado numérico, para convoluçao
    else:
        tipo_filtro = filtro            
        filtro = np.zeros((3,3), np.uint8) #matriz que serve para filtros de moda e mediana
    
    
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
        
    if tipo_filtro == 'media':
        resultx = int(((imx - filx + 2*padding)/desl) + 1)  #cálculo das dimensões da imagem convolucionada, para filtro numérico
        resulty = int(((imy - fily + 2*padding)/desl) + 1)
    
        resultado = np.zeros((resultx, resulty))
    
    else: #filtros classificatórios (moda/mediana) geram resultado de dimensões iguais da entrada contornada
        resultado = np.uint8(imgpad)
    
    for x in range(imx):  #iterando o filtro pela imagem - passo de cima para baixo
        if x > imx - filx: #caso o filtro vá a passar pelo limite de linhas - chegou no canto inferior direito termina convolução
            break
        
        if x % desl == 0:  #confirma que cada passo seguirá de acordo com a quantidade de deslocamento dado
            for y in range(imy): #da esquerda pra direita
                if y > imy - fily: #caso o filtro vá a passar pelo limite de colunas
                    break
                
                
                if y % desl == 0: #confirma que cada passo seguirá de acordo com a quantidade de deslocamento dado
                    if tipo_filtro == 'moda':
                        for i in range(filx):
                            for j in range(fily):
                                filtro[i][j] = resultado[i + x][j + y]
                        
                        vals, counts = np.unique(filtro,return_counts=True,axis=None)
                        moda_ind = np.argmax(counts)   
                        resultado[x + filx//2][y + fily//2] = vals[moda_ind] #filtro da moda: o pixel central correspondente ao filtro será atribuído, na imagem filtrada, o valor de moda dos valores lidos pelo filtro
                        
                    elif tipo_filtro == 'mediana':
                        for i in range(filx):
                            for j in range(fily):
                                filtro[i][j] = resultado[i + x][j + y]
                                    
                        resultado[x + filx//2][y + fily//2]  = np.median(filtro) #filtro da mediana: o pixel central correspondente ao filtro será atribuído, na imagem filtrada, a mediana dentre os valores lidos pelo filtro
                            
                    else:
                        if lim != None: #caso há limiar, aplica-se o filtro de média em relação ao limiar
                            if (filtro * imgpad[x: x + filx, y: y + fily]).sum() < lim:
                                resultado[x,y] = (filtro * imgpad[x: x + filx, y: y + fily]).sum()
                            else:
                                resultado[x,y] = imgpad[x,y]
                        else:
                            resultado[x,y] = (filtro * imgpad[x: x + filx, y: y + fily]).sum() #filtro da média sem limiar
                
    
    return np.uint8(resultado)


cv.imshow('original',img)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('original',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

filtro_media_3x3 = (1/9)*np.ones((3,3))

img_media = convolucao(img, filtro_media_3x3)

cv.imshow('filtro media 3x3',img_media)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro media 3x3',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

filtro_media = (1/25)*np.ones((5,5))

img_media = convolucao(img, filtro_media)

cv.imshow('filtro media 5x5',img_media)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro media 5x5',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

filtro_media = (1/49)*np.ones((7,7))

img_media = convolucao(img, filtro_media)

cv.imshow('filtro media 7x7',img_media)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro media 7x7',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

T=200

img_medialim = convolucao(img, filtro_media_3x3, lim=T)

cv.imshow('filtro media limiar T='+str(T),img_medialim)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro media limiar T='+str(T),960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

img_mediana = convolucao(img, 'mediana')

cv.imshow('filtro mediana',img_mediana)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro mediana',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada

img_moda = convolucao(img, 'moda')

cv.imshow('filtro moda',img_moda)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
cv.moveWindow('filtro moda',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada
#filtro da moda apresenta defeitos não identificados