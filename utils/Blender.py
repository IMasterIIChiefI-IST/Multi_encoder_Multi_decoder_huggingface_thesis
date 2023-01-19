from PIL import Image
import cv2
import numpy as np


def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def blend(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la, lb, mask in zip(laplacian_A, laplacian_B, mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS


def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


def Blender(ImOriginal, ImLabel , X1, Y1 , X2, Y2 ,Color ,BlurZone = None ,path = None, denoise = False, reverse = False):
    if BlurZone is None:
        BlurZone = 0.015
    img1 = np.array(ImOriginal)
    img1 = img1[..., ::-1].copy()
    img2 = Image.new(mode="RGB", size=(img1.shape[1], img1.shape[0]), color=Color)
    img2.paste(ImLabel, (X1, Y1, X2, Y2))
    img2 = np.array(img2)
    img2 = img2[..., ::-1].copy()

    if reverse:
        X11 = X1 + int((X2 - X1) * BlurZone)
        X21 = X2 - int((X2 - X1) * BlurZone)
        Y11 = Y1 + int((Y2 - Y1) * BlurZone)
        Y21 = Y2 - int((Y2 - Y1) * BlurZone)
        mask = np.ones((img2.shape[0], img2.shape[1], 3), dtype='float32')
        mask[Y11: Y21, X11: X21, :] = (0, 0, 0)
    else:
        X11 = X1 - int((X2 - X1) * BlurZone)
        X21 = X2 + int((X2 - X1) * BlurZone)
        Y11 = Y1 - int((Y2 - Y1) * BlurZone)
        Y21 = Y2 + int((Y2 - Y1) * BlurZone)
        mask = np.zeros((img2.shape[0], img2.shape[1], 3), dtype='float32')
        mask[Y11: Y21, X11: X21, :] = (1, 1, 1)

    num_levels = 4
    # For image-1, calculate Gaussian and Laplacian
    gaussian_pyr_1 = gaussian_pyramid(img1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    # For image-2, calculate Gaussian and Laplacian
    gaussian_pyr_2 = gaussian_pyramid(img2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = gaussian_pyramid(mask, num_levels)
    mask_pyr_final.reverse()
    # Blend the images
    if reverse:
        add_laplace = blend(laplacian_pyr_2, laplacian_pyr_1, mask_pyr_final)
    else:
        add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
    # Reconstruct the images
    final = reconstruct(add_laplace)
    # Save the final image to the disk
    if denoise:
        image = cv2.imread(path)
        image = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)
        cv2.imwrite(path, image)
    if path is None:
        return Image.fromarray(np.uint8(final[num_levels][..., ::-1].copy()))
    else:
        cv2.imwrite(path, final[num_levels])