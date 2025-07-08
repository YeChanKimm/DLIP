import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

kernel = np.ones((3, 5), np.uint8)


# 이미지를 가져온다.
source = cv.imread('Image/LV1.png', cv.IMREAD_COLOR)
source = cv.resize(source, (960, 540))

print("source shape:", source.shape)
if source is None:
    print("이미지를 불러올 수 없습니다.")
    exit()


#roi 설정

#소스 이미지를 선택한 후 엣지 출력
source_blue, source_green, source_red = cv.split(source)

v = np.median(source_red)


# lower, upper threshold 자동 계산
# lower = int(max(0,  1.08*v))
# upper = int(min(255, 2.16 * v))

#소스이미지에서 엣지 검출
source_edges = cv.Canny(source_red, 140, 220, apertureSize=3)

#====================roi========================

#roi를 만들기 위한 요소 선정
binary = source_edges.copy()
mask= np.zeros_like(binary)

#roi좌표
pts = np.array([[
    (3,255),  # 왼쪽 위
    (416,226),  # 오른쪽 위
    (285,500),  # 오른쪽 아래
    (94, 503)    # 왼쪽 아래
]], dtype=np.int32)


#선택한 좌표 내부만 흰색인 마스크 완성
cv.fillPoly(mask,pts,255)


#검출된 엣지와 선정한 부분만 흰색인 부분을 곱하여 roi설정
roi = cv.bitwise_and(mask,source_edges)

roi = cv.GaussianBlur(roi, (5,5), 5)
roi = cv.morphologyEx(roi, cv.MORPH_CLOSE, kernel)
roi = cv.morphologyEx(roi, cv.MORPH_OPEN, kernel)

#=======================contour======================

#컨투어 이미지를 넣기 위한 빈 행렬을 준비
no_filtered_contours_image = np.zeros_like(binary)
filtered_contours_image= np.zeros_like(binary)

#초기 roi이미지에 대해서 컨투어 찾기
contours_roi, _ = cv.findContours(roi, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(no_filtered_contours_image, contours_roi, -1, 255, 2)


# for cnt in contours_roi:
#     area=cv.contourArea(cnt)
#     x, y, w, h = cv.boundingRect(cnt)
#     print("area:", area)
#     print("width:", w)
#     print("height:", h)
#     print("\n")
    
#필터링된 컨투어를 담기위한 배열
contours_filtered = []

for cnt in contours_roi:
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)

    if w==0:
        continue
        
    if area >= 700 and w>70 and h/w>0.5:
        contours_filtered.append(cnt)

if not contours_filtered:
    print("필터링된 컨투어가 없습니다.")
    exit()

cv.drawContours(filtered_contours_image, contours_filtered, -1, 255, 2)

#=============================contour=============================

# 1. 컨투어 점들 모두 모으기
all_points = np.vstack(contours_filtered).squeeze()

# 2. x 기준 정렬
all_points = all_points[np.argsort(all_points[:, 0])]
x = all_points[:, 0]
y = all_points[:, 1]

# 3. polyfit으로 2차 곡선 근사
coeffs = np.polyfit(x, y, deg=2)
poly = np.poly1d(coeffs)

x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = poly(x_fit)

# 4. 원본 이미지에 곡선 그리기 (y 좌표 보정!)
roi_y_start = filtered_contours_image[:1].max()  # y offset
source_poly = source.copy()
for i in range(len(x_fit)-1):
    pt1 = (int(x_fit[i]), int(y_fit[i]+roi_y_start))  # y좌표에 roi offset 추가!
    pt2 = (int(x_fit[i+1]), int(y_fit[i+1]+roi_y_start))
    if 0 <= pt1[1] < source.shape[0] and 0 <= pt2[1] < source.shape[0]:
        cv.line(source_poly, pt1, pt2, (255, 0, 255), 2)  # 분홍색


#===========================printing======================
plt.figure()
plt.imshow(source_edges, cmap='gray')
plt.title('final')
plt.axis('off')


plt.figure()
plt.imshow(roi, cmap='gray')
plt.title('roi')
plt.axis('off')

plt.figure()
plt.imshow(no_filtered_contours_image, cmap='gray')
plt.title('no_filtered_contours_image')
plt.axis('off')


plt.figure()
plt.imshow(filtered_contours_image, cmap='gray')
plt.title('filtered_contours_image')
plt.axis('off')

bottom_image, width=source_poly.shape[:2]
threshold_1=bottom_image-250
threshold_2=bottom_image-120

print("bottom_image:",bottom_image)
cv.line(source_poly, (0,threshold_1), (width,threshold_1), (0 , 255 , 0) , 1 , cv.LINE_AA)
cv.line(source_poly, (0,threshold_2), (width,threshold_2), (0 , 0 , 255) , 1 , cv.LINE_AA)

plt.figure()
plt.imshow(source_poly)
plt.title('final')
plt.axis('off')


plt.show()