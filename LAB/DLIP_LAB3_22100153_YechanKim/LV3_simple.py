# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# kernel = np.ones((3, 15), np.uint8)

# # Load image
# source = cv.imread('Image/LV3_simple.png', cv.IMREAD_COLOR)
# if source is None:
#     print("이미지를 불러올 수 없습니다.")
#     exit()

# source_blue, source_green, source_red = cv.split(source)
# source_edges = cv.Canny(source_red, threshold1=80, threshold2=160, apertureSize=3)

# roi_y_start = 451  # y offset
# roi = source_edges[451:1020, 0:826]
# roi = cv.GaussianBlur(roi, (5,5), 5)

# binary = roi.copy()
# mask = np.zeros_like(binary)

# contours_roi, _ = cv.findContours(roi, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)

# filtered_contours = []
# for cnt in contours_roi:
#     area = cv.contourArea(cnt)
#     if area >= 1000:
#         filtered_contours.append(cnt)

# cv.drawContours(mask, filtered_contours, -1, 255, 2)

# # ---------------------- 🎯 곡선 근사 시작 ----------------------

# # 1. 컨투어 점들 모두 모으기
# all_points = np.vstack(filtered_contours).squeeze()

# # 2. x 기준 정렬
# all_points = all_points[np.argsort(all_points[:, 0])]
# x = all_points[:, 0]
# y = all_points[:, 1]

# # 3. polyfit으로 2차 곡선 근사
# coeffs = np.polyfit(x, y, deg=2)
# poly = np.poly1d(coeffs)

# x_fit = np.linspace(x.min(), x.max(), 1000)
# y_fit = poly(x_fit)

# # 4. 원본 이미지에 곡선 그리기 (y 좌표 보정!)
# source_poly = source.copy()
# for i in range(len(x_fit)-1):
#     pt1 = (int(x_fit[i]), int(y_fit[i] + roi_y_start))  # y좌표에 roi offset 추가!
#     pt2 = (int(x_fit[i+1]), int(y_fit[i+1] + roi_y_start))
#     if 0 <= pt1[1] < source.shape[0] and 0 <= pt2[1] < source.shape[0]:
#         cv.line(source_poly, pt1, pt2, (255, 0, 255), 2)  # 분홍색

# # ---------------------- 🎯 시각화 ----------------------

# plt.figure()
# plt.imshow(cv.cvtColor(source_poly, cv.COLOR_BGR2RGB))
# plt.title('근사된 곡선')
# plt.axis('off')

# plt.show()
