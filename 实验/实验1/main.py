import cv2
import sift
import ransac

path1 = '000005.jpg'
path2 = '000007.jpg'

img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)

# # 调包的
# sift = cv2.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

kp1, des1 = sift.computeKeypointsAndDescriptors(img1)
kp2, des2 = sift.computeKeypointsAndDescriptors(img2)

# FLANN 参数设计
match = cv2.BFMatcher()
matches = match.knnMatch(des1, des2, k=2)

# Apply ratio test
# 比值测试，首先获取与 A距离最近的点 B （最近）和 C （次近），
# 只有当 B/C 小于阀值时（0.75）才被认为是匹配，
# 因为假设匹配是一一对应的，真正的匹配的理想距离为0
good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append([m])

# print(good[0][0])
# print(len(kp1), len(kp2))
# print(type(kp1[good[0][0].queryIdx].pt))
# print(len(good))

Max_num, H, inlier_points = ransac.ransac(good, kp1, kp2, confidence=4, iter_num=500)
print(H)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inlier_points, None, flags=2)
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow("image1", img3)
cv2.imshow("image2", img4)
cv2.waitKey(0)  # 等待按键按下
cv2.destroyAllWindows()

Panorama = cv2.warpPerspective(img2, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))
cv2.imshow("Panorama", Panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 将左图加入到变换后的右图像的左端即获得最终图像
Panorama[0:img1.shape[0], 0:img1.shape[1]] = img1
cv2.namedWindow("Panorama", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Panorama", Panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
