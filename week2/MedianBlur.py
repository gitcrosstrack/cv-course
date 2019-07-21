# Finish 2D convolution/filtering by your self.
#
# What you are supposed to do can be described as "median blur", which means by using a sliding window
#
# on an image, your task is not going to do a normal convolution, but to find the median value within
#
# that crop.
#
# #
#
# You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#
# And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When
#
# "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#
# image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis
#
# depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
# #
#
# Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version
#
# with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#
# Follow up 1: Can it be completed in a shorter time complexity?
#
# #
#
# Python version:
#
# def medianBlur(img, kernel, padding_way):
#
# img & kernel is List of List; padding_way a string
#
# Please finish your code under this blank

import numpy as np
import cv2

#排序中值查找方法
def getMedian(img, kernel_width, kernel_height, i, j):
    data = []

    for m in range(kernel_width):
        for n in range(kernel_height):
            # print("%d %d %d %d" %(i,j,m,n))
            data.append(img[i + n][j + m])
    data.sort()
    return data[len(data) // 2]

#直方图中值查找方法
def getMedianByHist(img, kernel_width, kernel_height, i, j, hist, median, n):
    if j == 0:
        median = getMedian(img, kernel_width, kernel_height, i, j)
        for m in range(kernel_width):
            for p in range(kernel_height):
                cur_val = img[i + p][j + m]
                hist[cur_val] += 1
        n = sum(hist[:median + 1])
        return (median, n)

    standard_n = kernel_width * kernel_height // 2
    for p in range(kernel_height):
        cur_val = img[i + p][j - 1]
        hist[cur_val] -= 1
        if cur_val <= median:
            n -= 1
    for p in range(kernel_height):
        cur_val = img[i + p][j + kernel_width - 1]
        hist[cur_val] += 1
        if cur_val <= median:
            n += 1
    # print('--------new-----')
    if n > standard_n:
        while n > standard_n and median >= 0:
            # print('----')

            n -= hist[median]
            median -= 1
    if n < standard_n:
        while n < standard_n:
            # print('++++')
            median += 1
            n += hist[median]

    n = sum(hist[:median + 1])
    # print(hist)

    return (median, n)


def medianBlur(img, kernel, padding_way='ZERO'):
    ways = ['ZERO', 'REPLICA']
    if padding_way not in ways:
        raise Exception('padding_way invalid')

    kernel_width = len(kernel[0])
    kernel_height = len(kernel)
    img_width = len(img[0])
    img_height = len(img)
    padding_up = kernel_height // 2
    padding_down = kernel_height - 1 - padding_up
    padding_left = kernel_width // 2
    padding_right = kernel_width - 1 - padding_left

    padding_img_width = kernel_width + img_width - 1
    padding_img_height = kernel_height + img_height - 1

    if padding_way == 'ZERO':
        # 填充 先把两侧的填充 【000left000】【original img】 + 【000right000】
        padding_img = [[0 for j in range(padding_left)] + list(row) + [0 for j in range(padding_right)] for row in img]
        # 填充top 和 bottom
        padding_img = [[0 for j in range(padding_img_width)] for i in range(padding_up)] + padding_img + [
            [0 for j in range(padding_img_width)] for i in range(padding_down)]
    else:
        # 填充 先把两侧的填充 【***left***】【original img】 + 【***right***】
        padding_img = [[row[0] for j in range(padding_left)] + list(row) + [row[-1] for j in range(padding_right)] for
                       row in img]
        # 填充top 和 bottom 【000left000】 img【0】+【000right000】
        padding_img = [[0 for j in range(padding_left)] + list(img[0]) + [0 for j in range(padding_right)] for i in
                       range(padding_up)] + padding_img + [
                          [0 for j in range(padding_left)] + list(img[-1]) + [0 for j in range(padding_right)] for i in
                          range(padding_down)]

    img_out = []
    for i in range(img_height):
        row = []
        for j in range(img_width):
            row.append(getMedian(padding_img, kernel_width, kernel_height, i, j))
        img_out.append(row)
    return img_out


def fastMedianBlur(img, kernel, padding_way='ZERO'):
    ways = ['ZERO', 'REPLICA']
    if padding_way not in ways:
        raise Exception('padding_way invalid')

    kernel_width = len(kernel[0])
    kernel_height = len(kernel)
    img_width = len(img[0])
    img_height = len(img)
    padding_up = kernel_height // 2
    padding_down = kernel_height - 1 - padding_up
    padding_left = kernel_width // 2
    padding_right = kernel_width - 1 - padding_left

    padding_img_width = kernel_width + img_width - 1
    padding_img_height = kernel_height + img_height - 1

    if padding_way == 'ZERO':
        # 填充 先把两侧的填充 【000left000】【original img】 + 【000right000】
        padding_img = [[0 for j in range(padding_left)] + list(row) + [0 for j in range(padding_right)] for row in img]
        # 填充top 和 bottom
        padding_img = [[0 for j in range(padding_img_width)] for i in range(padding_up)] + padding_img + [
            [0 for j in range(padding_img_width)] for i in range(padding_down)]
    else:
        # 填充 先把两侧的填充 【***left***】【original img】 + 【***right***】
        padding_img = [[row[0] for j in range(padding_left)] + list(row) + [row[-1] for j in range(padding_right)] for
                       row in img]
        # 填充top 和 bottom 【000left000】 img【0】+【000right000】
        padding_img = [[0 for j in range(padding_left)] + list(img[0]) + [0 for j in range(padding_right)] for i in
                       range(padding_up)] + padding_img + [
                          [0 for j in range(padding_left)] + list(img[-1]) + [0 for j in range(padding_right)] for i in
                          range(padding_down)]

    img_out = []

    for i in range(img_height):
        row = []
        # get median
        median = 0
        n = 0
        # init hist
        hist = [0 for i in range(256)]
        # cal numbers lower than median
        n = sum([val for val in hist if val <= median])
        for j in range(0, img_width):
            (median, n) = getMedianByHist(padding_img, kernel_width, kernel_height, i, j, hist, median, n)
            row.append(median.astype('uint8'))
        img_out.append(row)
    return img_out

def make_noise(img):
    '''
    添加椒盐噪声
    '''

    newimg=np.array(img)
    #噪声点数量
    noisecount=50000
    for k in range(0,noisecount):
        xi=int(np.random.uniform(0,newimg.shape[1]))
        xj=int(np.random.uniform(0,newimg.shape[0]))
        newimg[xj,xi]=255
    return newimg



if __name__ == "__main__":
    img_uri = 'lbxx.jpg'
    img = cv2.imread(img_uri, 0)
    img = make_noise(img)
    kernel = [[1, 2, 3], [2, 4, 6], [3, 2, 1]]
    fast_median_img = fastMedianBlur(list(img), kernel, 'REPLICA')
    slow_median_img = medianBlur(list(img), kernel, 'REPLICA')
    cv2.imshow('median_fast', np.array(fast_median_img))
    cv2.imshow('median_slow', np.array(fast_median_img))
    cv2.imshow('origin', img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
