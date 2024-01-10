import numpy as np
import cv2
import math


# 卷积函数
def convolution(image, kernel, average=False):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_blur(image, size):
    sigma = math.sqrt(size)
    # 获取高斯滤波核
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel = np.outer(kernel_1D.T, kernel_1D.T)
    kernel *= 1.0 / kernel.max()
    return convolution(image, kernel, average=True)


# 通过Sobel检测获取梯度赋值及方向
def sobel_detect(image):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gx = convolution(image, kernel)
    Gy = convolution(image, np.flip(kernel.T, axis=0))

    gradient_magnitude = np.sqrt(np.square(Gx) + np.square(Gy))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    gradient_direction = np.arctan2(Gy, Gx)
    gradient_direction = np.rad2deg(gradient_direction)
    gradient_direction += 180

    return gradient_magnitude, gradient_direction

# 非极大值抑制
def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    return output


def threshold(image, low, high, weak):
    output = np.zeros(image.shape)

    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    return output


def hysteresis(image, weak):
    image_row, image_col = image.shape

    top_to_bottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[
                        row - 1, col] == 255 or top_to_bottom[
                        row + 1, col] == 255 or top_to_bottom[
                        row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[
                        row - 1, col + 1] == 255 or top_to_bottom[
                        row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[
                        row - 1, col] == 255 or bottom_to_top[
                        row + 1, col] == 255 or bottom_to_top[
                        row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[
                        row - 1, col + 1] == 255 or bottom_to_top[
                        row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[
                        row - 1, col] == 255 or right_to_left[
                        row + 1, col] == 255 or right_to_left[
                        row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[
                        row - 1, col + 1] == 255 or right_to_left[
                        row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[
                        row - 1, col] == 255 or left_to_right[
                        row + 1, col] == 255 or left_to_right[
                        row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[
                        row - 1, col + 1] == 255 or left_to_right[
                        row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255

    return final_image


def canny(image):
    gradient_magnitude, gradient_direction = sobel_detect(image)
    image = non_max_suppression(gradient_magnitude, gradient_direction)
    weak = 50
    image = threshold(image, 5, 20, weak=weak)
    new_image = hysteresis(image, weak)
    return new_image

# 生成mask掩膜，输入为宽、高，读取mask.png实现
def generate_mask(w, h):
    mask_img = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.resize(mask_img, (w, h))
    _, mask = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)
    return mask


def ROI(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

# Hough直线检测
def lines_detector_hough(edge, ThetaDim=None, DistStep=None, threshold=None, halfThetaWindowSize=2,
                         halfDistWindowSize=None):
    row, col = edge.shape
    if ThetaDim is None:
        ThetaDim = 90
    if DistStep is None:
        DistStep = 1
    # 计算距离分段数量
    MaxDist = np.sqrt(row ** 2 + col ** 2)
    DistDim = int(np.ceil(MaxDist / DistStep))
    if halfDistWindowSize is None:
        halfDistWindowSize = int(DistDim / 50)
    # 建立投票
    accumulator = np.zeros((ThetaDim, DistDim))  # theta的范围是[0,pi). 在这里将[0,pi)进行了线性映射.类似的,也对Dist轴进行了线性映射
    #
    sinTheta = [np.sin(t * np.pi / ThetaDim) for t in range(ThetaDim)]
    cosTheta = [np.cos(t * np.pi / ThetaDim) for t in range(ThetaDim)]
    # 计算距离（rho）
    for i in range(row):
        for j in range(col):
            if not edge[i, j] == 0:
                for k in range(ThetaDim):
                    accumulator[k][int(round((i * cosTheta[k] + j * sinTheta[k]) * DistDim / MaxDist))] += 1
    M = accumulator.max()
    # ---------------------------------------
    # 非极大抑制
    if threshold is None:
        threshold = int(M * 1.369 / 10)
    result = np.array(np.where(accumulator > threshold))  # 阈值化
    # 获得对应的索引值
    temp = [[], []]
    for i in range(result.shape[1]):
        eight_neiborhood = accumulator[
                           max(0, result[0, i] - halfThetaWindowSize + 1):min(result[0, i] + halfThetaWindowSize,
                                                                              accumulator.shape[0]),
                           max(0, result[1, i] - halfDistWindowSize + 1):min(result[1, i] + halfDistWindowSize,
                                                                             accumulator.shape[1])]
        if (accumulator[result[0, i], result[1, i]] >= eight_neiborhood).all():
            temp[0].append(result[0, i])
            temp[1].append(result[1, i])
    # 记录原图所检测的坐标点（x,y）
    result_temp = np.array(temp)
    # -------------------------------------------------------------
    result = result_temp.astype(np.float64)
    result[0] = result[0] * np.pi / ThetaDim
    result[1] = result[1] * MaxDist / DistDim
    return result, result_temp

# 画线
def drawLines(lines, image, mask, color=(0, 255, 255), err=3):
    result = image
    Cos = np.cos(lines[0])
    Sin = np.sin(lines[0])
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            e = np.abs(lines[1] - i * Cos - j * Sin)
            if (e < err).any() and mask[i, j]:
                result[i, j] = color
    return result


def process_image(image, mask):
    # 灰度转换
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 滤波去噪
    blurred_image = gaussian_blur(gray, 3)
    blurred_image = np.uint8(blurred_image)
    # Canny边缘检测
    edges = canny(blurred_image)
    # 提取ROI
    masked_edge = ROI(edges, mask)
    # 二值化
    _, edge = cv2.threshold(masked_edge, 128, 255, cv2.THRESH_BINARY)
    # Hough直线检测
    lines, _ = lines_detector_hough(edge)
    # 画线，得到结果
    result = drawLines(lines, image, mask)
    return result

#
# if __name__ == "__main__":
#     img_path = 'color.png'
#     img = cv2.imread(img_path)
#     mask = generate_mask(img)
#     process_image(img, mask)
