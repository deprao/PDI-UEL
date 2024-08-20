import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('gato-vaquero-emoji.png')

#img.shape[0] é a altura e img.shape[1] a largura

imht = img.shape[0]
imlen = img.shape[1]

print("A imagem lida possui dimensões de pixels "+str(imlen)+"x"+str(imht)+" largura X altura.\n")
samples = input("Insira a quantidade de amostras de resolução reduzida desejadas: ")
samples = float(samples)
sample_factor = input("Insira um fator de amostragem - por quanto gostaria que a imagem fosse reduzida por amostra\n(exemplo: 2 reduzirá cada uma pela metade): ")
sample_factor = int(sample_factor)

#alocação de espaço para conjunto de subplots
if int(samples) % int(samples) == 0 :
    grid_shape = str(int(np.ceil(np.sqrt(samples))))+str(int(np.ceil(np.sqrt(samples)))+1)+"1"
else :
     grid_shape = str(int(np.ceil(np.sqrt(samples))))+str(int(np.ceil(np.sqrt(samples))))+"1"

subplot_grid = int(grid_shape) #posição do primeiro subplot comparativo de resolução, de acordo com o número de reduções dado
print(subplot_grid)

samples = int(samples)
smaller_ht = int(imht/(sample_factor**(samples+1)))  #medidas de resolução após a menor desejada, como condição de parada
smaller_len = int(imlen/(sample_factor**(samples+1)))

while True : #loop até a menor razão de tamanho de acordo com os parâmetros
    plt.subplot(subplot_grid),plt.imshow(img),plt.title(str(imlen)+'x'+str(imht)) #adiciona um subplot ao conjunto comparativo lado a lado, com medidas largura X altura
    #conjunto de subplots pode não gerar por tamanho/quantidade das amostras
    
    cv.imshow('amostragem',img)         #mostra cada amostra de imagem na tela, começando com a inicial padrão
    cv.moveWindow('amostragem',960,540) #centraliza a âncora de amostra de imagem - neste caso para uma tela 1920x1080 - deve-se trocar os respectivos (x,y) se precisar
    cv.waitKey(0) #pausa amostra na tela, esperando por qualquer input do teclado ou ser fechada
    
    len_new = int(imlen/sample_factor)
    ht_new = int(imht/sample_factor)
    
    img_lower = np.zeros((imht,imlen,3), np.uint8) #amostragem temporária de primeiro nível de rebaixamento de resolução a partir de matriz de zeros

    i = 0
    l = 0
    for j in range(0,(ht_new-1)) :
        for k in range(0,(len_new-1)) :     #preenche a matriz da imagem rebaixada
            img_lower[j][k] = img[i][l]
            img_lower[j][k+1] = img[i][l]
            img_lower[j+1][k] = img[i][l]
            l += sample_factor
        img_lower[j+1][k+1] = img[i][l]
        l = 0
        i += sample_factor
            
            
    img = img_lower
    subplot_grid += 1  #avança uma posição no conjunto de subplots, para colocar a imagem recém criada temporariamente
    imht = ht_new
    imlen = len_new
    if imht == smaller_ht and imlen == smaller_len :
        break
    
plt.show(),plt.title('amostragens lado a lado') #matplotlib mostrando lado a lado, por padrão troca o mapeamento de cores