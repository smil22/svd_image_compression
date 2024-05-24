import numpy as np
import matplotlib.pyplot as plt


def svd_image_compression(img_pwd,compression_rate):
    """
    It compresses the given image and show it.
    - img_pwd: image file path
    - compression_rate: you must give it in percentage format (Ex: 60,50,20,...)
    """
    img = plt.imread(img_pwd)
    n,m = img.shape[0],img.shape[1]
    A = np.asarray(img)
    A = A[:,:,0]
    U,s,V=np.linalg.svd(A,full_matrices=False)
    S = np.diag(s)
    k = m * (1-compression_rate/100)
    k = int(k)
    Ak = np.dot(U[:,:k],S[:k,:k]).dot(V[:k,:])
    plt.imshow(Ak,cmap=plt.cm.gray)
    plt.title('Taux de compression = {0:2.2f}   Nbre de Pixels = {1:}'.format(compression_rate,n*k))
    plt.axis('off')
    return Ak
