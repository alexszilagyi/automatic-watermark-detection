import cv2
import numpy as np

KERNEL_SIZE = 3


def estimate_watermark(images):
    """
    Given a folder, estimate the watermark (grad(W) = median(grad(J)))
    Also, give the list of gradients, so that further processing can be done on it
    """
    print("Computing gradients.")
    gradx = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), images))
    grady = list(map(lambda x: cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE), images))

    print("Computing median gradients.")
    Wm_x = np.median(np.array(gradx), axis=0)
    Wm_y = np.median(np.array(grady), axis=0)

    return Wm_x, Wm_y


def poisson_reconstruct(gradx, grady, kernel_size=KERNEL_SIZE, num_iters=100, h=0.1,
                        boundary_image=None, boundary_zero=True):
    """
    Iterative algorithm for Poisson reconstruction.
    Given the gradx and grady values, find laplacian, and solve for image
    Also return the squared difference of every step.
    h = convergence rate
    """
    fxx = cv2.Sobel(gradx, cv2.CV_64F, 1, 0, ksize=kernel_size)
    fyy = cv2.Sobel(grady, cv2.CV_64F, 0, 1, ksize=kernel_size)
    laplacian = fxx + fyy
    m, n, p = laplacian.shape

    if boundary_zero:
        est = np.zeros(laplacian.shape)
    else:
        assert (boundary_image is not None)
        assert (boundary_image.shape == laplacian.shape)
        est = boundary_image.copy()

    est[1:-1, 1:-1, :] = np.random.random((m - 2, n - 2, p))
    loss = []

    for i in range(num_iters):
        old_est = est.copy()
        est[1:-1, 1:-1, :] = 0.25 * (est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] + est[2:, 1:-1, :] + est[1:-1, 2:, :]
                                     - h * h * laplacian[1:-1, 1:-1, :])
        error = np.sum(np.square(est - old_est))
        loss.append(error)

    return est


def PlotImage(image):
    """
    PlotImage: Give a normalized image matrix which can be used with implot, etc.
    Maps to [0, 1]
    """
    im = image.astype(float)
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def image_threshold(image, threshold=0.5):
    '''
    Threshold the image to make all its elements greater than threshold*MAX = 1
    '''
    m, M = np.min(image), np.max(image)
    im = PlotImage(image)
    im[im >= threshold] = 1
    im[im < 1] = 0
    return im


def crop_watermark(gradx, grady, threshold=0.4, boundary_size=2):
    """
    Crops the watermark by taking the edge map of magnitude of grad(W)
    Assumes the gradx and grady to be in 3 channels
    @param: threshold - gives the threshold param
    @param: boundary_size - boundary around cropped image
    """
    W_mod = np.sqrt(np.square(gradx) + np.square(grady))
    W_mod = PlotImage(W_mod)
    W_gray = image_threshold(np.average(W_mod, axis=2), threshold=threshold)
    x, y = np.where(W_gray == 1)

    xm, xM = np.min(x) - boundary_size - 1, np.max(x) + boundary_size + 1
    ym, yM = np.min(y) - boundary_size - 1, np.max(y) + boundary_size + 1

    return gradx[xm:xM, ym:yM, :], grady[xm:xM, ym:yM, :]


def watermark_detector(img, gx, gy, thresh_low=200, thresh_high=220, printval=False):
    """
    Compute a verbose edge map using Canny edge detector, take its magnitude.
    Assuming cropped values of gradients are given.
    Returns image, start and end coordinates
    """
    Wm = (np.average(np.sqrt(np.square(gx) + np.square(gy)), axis=2))

    img_edgemap = (cv2.Canny(img, thresh_low, thresh_high))
    chamfer_dist = cv2.filter2D(img_edgemap.astype(float), -1, Wm)

    rect = Wm.shape
    index = np.unravel_index(np.argmax(chamfer_dist), img.shape[:-1])
    if printval:
        print(index)

    x, y = (index[0] - rect[0] // 2), (index[1] - rect[1] // 2)
    return (x, y), (rect[0], rect[1])
