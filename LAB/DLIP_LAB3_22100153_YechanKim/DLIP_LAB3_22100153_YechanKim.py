import numpy as np
import cv2 as cv

cap = cv.VideoCapture('Image/LAB3_Video.mp4')
if not cap.isOpened():
    print("Error: 비디오 파일을 열 수 없습니다.")
    exit()

#모폴로지를 위한 커널을 설정
kernel = np.ones((3, 5), np.uint8)

while True:

    #비디오 불러오기
    ret, source = cap.read()

    if not ret:
        break
    
    
    #=================================edge========================================
    source_blue, source_green, source_red = cv.split(source)

    # red 채널 기준 median 계산
    v = np.median(source_red)


    # lower, upper threshold 계산
    lower = int(max(0,  0.90*v))
    upper = int(min(255, 1.80*v))

    source_edges = cv.Canny(source_red, lower, upper)
   #==================================roi========================================

    
    

    #roi좌표
    pts = np.array([[ 
        (3, 468), 	#좌측상단
        (790, 380), #우측상단
        (495, 1080),#좌측하단
        (3, 1080) 	#우측하단
		]], dtype=np.int32)

    #선택한 좌표 내부만 흰색인 마스크 완성
    binary = source_edges.copy()
    mask= np.zeros_like(binary)
    cv.fillPoly(mask,pts,255)

    #검출된 엣지와 선정한 부분만 흰색인 부분을 곱하여 roi설정
    roi = cv.bitwise_and(mask,source_edges)

    #roi에 대한 필터링, 모폴로지
    roi = cv.GaussianBlur(roi, (5, 5), 5)
    roi = cv.morphologyEx(roi, cv.MORPH_CLOSE, kernel)


    #===============================contour=======================================
    
    #컨투어 이미지를 넣기 위한 빈 행렬을 준비
    no_filtered_contours_image = np.zeros_like(binary)
    filtered_contours_image = np.zeros_like(binary)
    
    #초기 roi이미지에 대해서 컨투어 찾기
    contours_roi, _ = cv.findContours(roi, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(no_filtered_contours_image, contours_roi, -1, 255, 2)


    # 초기 roi이미지에 대해서 컨투어 찾기
    contours_roi, _ = cv.findContours(roi, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(no_filtered_contours_image, contours_roi, -1, 255, 2)

  
    #필터링된 컨투어를 저장할 공간
    contours_filtered = []

    for cnt in contours_roi:
        
        #컨투어별 면적 계산
        area = cv.contourArea(cnt)

        #외접원의 좌표, 너비, 높이
        x, y, w, h = cv.boundingRect(cnt)
        
        #높이에 대한 너비의 비율
        ratio=h/w

        #분모가 0인 상황 방지
        if w == 0:
            continue
        
        #곡선의 기준을 만족하는 컨투어를 새로 저장
        if (area >= 1300)  or (600<=area<1300 and ratio>0.7):
            contours_filtered.append(cnt)
    
    #필터링된 컨투어가 없을 경우 예외처리
    if not contours_filtered:
        continue    
    #===============================fitting=========================


    #필터링된 컨투어 점들을 모두 모으기
    all_points = np.vstack(contours_filtered).squeeze()

    #x축 기준 정렬, x, y 정의
    all_points = all_points[np.argsort(all_points[:, 0])]
    x = all_points[:, 0]
    y = all_points[:, 1]

    #2차함수형태로 곡선 근사
    coeffs = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coeffs)
    
    #fitting된 함수에 대입할 x,y값 정의
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly(x_fit)

   
    
    #피팅된 곡선을 그릴 행렬 정의
    final_line=np.zeros_like(binary)
    final_line=cv.cvtColor(final_line,cv.COLOR_GRAY2BGR)
    
    #빈 이미지에 피팅된 곡선 그리기
    for i in range(len(x_fit) - 1):
        pt1 = (int(x_fit[i]), int(y_fit[i]))
        pt2 = (int(x_fit[i + 1]), int(y_fit[i + 1]))
        if 0 <= pt1[1] < source.shape[0] and 0 <= pt2[1] < source.shape[0]:
            cv.line(final_line, pt1, pt2, (0, 255, 0), 2)
    
    #소스 이미지와 피팅된 곡선 합치기
    final_output = cv.addWeighted(source, 1.0, final_line, 1.0, 0)
    
    #예외처리
    if not contours_filtered:
        cv.putText(final_output,
                'No contours found',
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA)
        cv.imshow('final', final_output)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
        continue
    
    
    #===============================score=======================================
    
    #최종 라인 이미지 내에서 0이 아닌 부분 가져오기
    final_line=cv.cvtColor(final_line,cv.COLOR_BGR2GRAY)
    points_final_line=cv.findNonZero(final_line)
   

    #가장 큰 y좌표가 이미지 하단에서 떨어진 정도를 score로 삼음
    bottom_image, width=final_output.shape[:2]
    ys = points_final_line[:, 0, 1]  
    score = bottom_image-np.max(ys)

    
    #스코어에 대한 레벨 설정
    if score>250:
        level=1
    elif score<=250 and score>=120:
        level=2
    else:
        level=3
    
    #레벨별 y좌표
    threshold_1 = int(bottom_image-250)
    threshold_2 = int(bottom_image-120)

    #레벨에 따라 점선
    for y_val, color in zip([threshold_1, threshold_2], [(0, 255, 0), (255, 255, 0)]):
        for x_pos in range(0, width, 30):
            cv.line(final_output, (x_pos, y_val), (x_pos + 10, y_val), color, 2)
    
    # ===========================print==========================

    #스코어와 레벨 텍스트를 담기 위한 사각형 설정
    box_x, box_y, box_w, box_h = 1400, 350, 250, 100
    cv.rectangle(final_output, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)

    # 스코어 텍스트 입력
    cv.putText(final_output,
               f'Score : {score:.2f}',
               (box_x + 10, box_y + 40),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2,
               cv.LINE_AA)

    # 레벨 텍스트 입력
    cv.putText(final_output,
               f'Level : {level}',
               (box_x + 10, box_y + 85),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2,
               cv.LINE_AA)

    # 노트북에서 1080*1920프레임의 영상이 다 담기지 않아 축소
    display_scale = 0.5 

    display_frame = cv.resize(final_output, (int(width * display_scale), int(bottom_image * display_scale)))
    cv.imshow('final output', display_frame)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
