import numpy as np
import cv2 as cv


img = cv.imread("abraxas.jpg", cv.IMREAD_GRAYSCALE)

cv.imshow('original',img)         #amostra de imagem inicial
cv.moveWindow('original',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)

EEst = cv.getStructuringElement(cv.MORPH_RECT, (5,5)) #elemento estruturante 5x5 prrenchido por 1

def complemento(imagem): #retorna o complemento de uma imagem
    full = np.zeros((imagem.shape[0],imagem.shape[1]),np.uint8)
    complemento = full - imagem
    
    return complemento

def abrir(imagem, ElementoEstruturante): #realiza abertura 
    erosao = cv.erode(imagem, ElementoEstruturante, iterations=1) #A⊝B
    abertura = cv.dilate(erosao, ElementoEstruturante, iterations=1)  #(A⊝B)⊕B

    return abertura

def fechar(imagem, ElementoEstruturante): #realiza fechamento
    dilatacao = cv.dilate(imagem, ElementoEstruturante, iterations=1) #A⊕B
    fechamento = cv.erode(dilatacao, ElementoEstruturante, iterations=1)#(A⊕B)⊝B
    
    return fechamento

def extrair_fronteiras(imagem, ElementoEstruturante): #extração de fronteiras
    erosao = cv.erode(imagem, ElementoEstruturante, iterations=1) #A⊝B
    fronteiras = imagem - erosao  #A - (A⊝B)
    
    return fronteiras

#preenchimento de buracos e extração de componentes conexos falhos: trabalhos acumularão tempos faltarão :^(

def preencher_buracos(imagem, ElementoEstruturante):
    comp = complemento(imagem)
    borda_img = extrair_fronteiras(img, ElementoEstruturante)
    img_interno = img - borda_img
    
    dilatar_buracos = cv.dilate(img_interno, ElementoEstruturante, iterations=1)
    
    img_dims = (img.shape[0], img.shape[1])
    comp_1d = np.array(comp, np.uint8).flatten()
    dilatar_buracos_1d = np.array(dilatar_buracos, np.uint8).flatten()
    preenchimento_1d = np.intersect1d(comp_1d, dilatar_buracos_1d)
    
    preenchimento = np.reshape(preenchimento_1d, img_dims)
    
    return preenchimento

def extrair_conexos(imagem, ElementoEstruturante):
    borda_img = extrair_fronteiras(img, ElementoEstruturante)
    img_interno = img - borda_img
    
    dilatar_conexos = cv.dilate(img_interno, ElementoEstruturante, iterations=1)
    
    img_dims = (img.shape[0], img.shape[1])
    img_1d = np.array(img, np.uint8).flatten()
    dilatar_conexos_1d = np.array(dilatar_conexos, np.uint8).flatten()
    conexos_1d = np.intersect1d(img_1d, dilatar_conexos_1d)
    
    conexos = np.reshape(conexos_1d, img_dims)
    
    return conexos


img_aberta = abrir(img, EEst)

cv.imshow('abertura',img_aberta)         #amostra de imagem inicial
cv.moveWindow('abertura',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)

img_fechada = fechar(img, EEst)

cv.imshow('fechamento',img_fechada)         #amostra de imagem inicial
cv.moveWindow('fechamento',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)

fronteiras_img = extrair_fronteiras(img, EEst)

cv.imshow('extr. de fronteiras',fronteiras_img)         #amostra de imagem inicial
cv.moveWindow('extr. de fronteiras',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)

img_preenchida = preencher_buracos(img, EEst)

cv.imshow('preenchimento de buracos',img_preenchida)         #amostra de imagem inicial
cv.moveWindow('preenchimento de buracos',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)

conexos_img = extrair_conexos(img, EEst)

cv.imshow('extr. de componentes conexos',conexos_img)         #amostra de imagem inicial
cv.moveWindow('extr. de componentes conexos',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
cv.waitKey(0)