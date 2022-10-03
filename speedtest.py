from array import array
from math import ceil, sqrt
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max as plm
from skimage.transform import radon

import cv2


def motion_blur(image: array, degree: int, angle: float) -> array:
    image = np.array(image)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(
        motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blur_img = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blur_img, blur_img, 0, 255, cv2.NORM_MINMAX)
    blur_img = np.array(blur_img, dtype=np.uint8)
    return blur_img


#   def axis_grad(image, step: int) -> tuple:
#       if type(image) == type([]):
#           x = np.shape(image)[2]
#           y = np.shape(image)[1]
#           x_axis = np.arange(0, x, x / step - 1)
#           y_axis = np.arange(0, y, y / step - 1)
#       else:
#           x = np.shape(image)[0]
#           y = np.shape(image)[1]
#           x_axis = np.arange(0, x, x / step - 1)
#           y_axis = np.arange(0, y, y / step - 1)
#       return (x_axis, y_axis)
 
def myshow(num: int, image, title) -> None:
    if num == 1:
        plt.subplot(111), plt.imshow(image, cmap='gray')
        plt.title(title), plt.xticks(), plt.yticks()
    else:
        for i in range(num):
            plt.subplot(ceil(sqrt(num)) * 10 + (round(num / sqrt(num))+1)
                        * 100 + i + 1), plt.imshow(image[i], cmap='gray')
            plt.title(title[i]), plt.xticks(), plt.yticks()
    return None

def my_fft(image: array,if_mf: int) -> array:
    img_f32 = np.float32(image)
    dft = cv2.dft(img_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fft_image = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    if if_mf == 1:
        fft_image = cv2.medianBlur(fft_image, 3)
    return fft_image


img = cv2.imread("test.jpeg", 0)
fft = my_fft(img, 1)
img_blur = motion_blur(img, 15, 0)
fft_blur = my_fft(img_blur, 1)

plt.imshow(img_blur,cmap = 'gray'),plt.show()


R_img = radon(fft, theta=np.arange(180))
R_fft = radon(fft_blur, theta=np.arange(180))


R_fft_show = cv2.resize(R_fft, dsize=(
    np.shape(img_blur)[0], np.shape(img_blur)[1]))

max_loc = plm(R_fft, min_distance=10, num_peaks=1)
for i in range(len(max_loc)):
    print(max_loc[i], R_fft[max_loc[i][0], max_loc[i][1]])
    plt.subplot(337+i),plt.plot(R_fft[max_loc][i][0]),plt.title('row with maxinum')

myshow(6, [img, fft, img_blur, fft_blur, R_img.T, R_fft.T, ], ['Input Image', 'FFT',
       'Blur', 'Blur FFT', 'Radon FFT', 'Radon Blur FFT'])
plt.show()





