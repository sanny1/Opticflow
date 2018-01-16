import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def main():
    gaus_sigma = 1

    frame_1 = cv2.imread('frame1.jpg',0)
    frame_2 = cv2.imread('frame2.jpg',0)

    frame_1 = cv2.normalize(frame_1.astype('float'),None,0.0,1.0,cv2.NORM_MINMAX)
    frame_2 = cv2.normalize(frame_2.astype('float'),None,0.0,1.0,cv2.NORM_MINMAX)

    Kernel_Size = 6*gaus_sigma+1
    k = (Kernel_Size-1)/2
    gaus_kernel_x = np.zeros((Kernel_Size,Kernel_Size))
    gaus_kernel_y = np.zeros((Kernel_Size,Kernel_Size))
    kernel = np.zeros((Kernel_Size,Kernel_Size))

    for i in range(Kernel_Size):
        for j in range(Kernel_Size):
            gaus_kernel_x[i,j] = -(((j-k-1)/(2*math.pi*(gaus_sigma**3)))*math.exp(-
                                  ((i-k-1)**2 + (j-k-1)**2)/(2*(gaus_sigma**2))))

    for i in range(Kernel_Size):
        for j in range(Kernel_Size):
            gaus_kernel_y[i,j] = -(((i-k-1)/(2*math.pi*(gaus_sigma**3)))*math.exp(-
                                  ((i-k-1)**2 + (j-k-1)**2)/(2*(gaus_sigma**2))))

    Dx_1 = cv2.filter2D(frame_1,-1,gaus_kernel_x)
    Dx_2 = cv2.filter2D(frame_2,-1,gaus_kernel_x)
    Dy_1 = cv2.filter2D(frame_1,-1,gaus_kernel_y)
    Dy_2 = cv2.filter2D(frame_2,-1,gaus_kernel_y)


    Ix = (Dx_1 + Dx_2)/2
    Iy = (Dy_1 + Dy_2)/2

    for i in range(Kernel_Size):
        for j in range(Kernel_Size):
            kernel[i,j] = ((1/(2*math.pi*(gaus_sigma**2)))*
                            math.exp(-(((i-k-1)**2 + (j-k-1)**2)/(2*(gaus_sigma**2)))))

    frame_1_s = cv2.filter2D(frame_1,-1,kernel)
    frame_2_s = cv2.filter2D(frame_2,-1,kernel)

    It = frame_2_s - frame_1_s

    n_s = 5
    h,w= frame_1.shape
    output_1 = np.zeros((h,w))
    output_2 = np.zeros((h,w))

    for i in np.arange((1+np.floor(n_s/2)),(h-np.floor(n_s/2))):
        for j in np.arange((1+np.floor(n_s/2)),(w-np.floor(n_s/2))):
            i = int(i)
            j = int(j)
            A = np.zeros((2,2))
            B = np.zeros((2,1))

            for m in np.arange(i-np.floor(n_s/2),i+np.floor(n_s/2)):
                for n in np.arange(j-np.floor(n_s/2),j+np.floor(n_s/2)):
                    m = int(m)
                    n = int(n)
                    A[0,0] = A[0,0] + (Ix[m,n]*Ix[m,n])
                    A[0,1] = A[0,1] + (Ix[m,n]*Iy[m,n])
                    A[1,0] = A[1,0] + (Ix[m,n]*Iy[m,n])
                    A[1,1] = A[1,1] + (Iy[m,n]**2)

                    B[0,0] = B[0,0] + (Ix[m,n]*It[m,n])
                    B[1,0] = B[1,0] + (Iy[m,n]*It[m,n])

            inv_A = np.linalg.pinv(A)
            result = np.matmul(inv_A,(-B))

            output_1[i,j] = result[0,0]
            output_2[i,j] = result[1,0]

    output_1 = np.flipud(output_1)
    output_2 = np.flipud(output_2)

    plt.quiver(output_1,output_2)
    plt.show()

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
