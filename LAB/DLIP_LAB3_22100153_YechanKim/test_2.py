import numpy as np
import cv2 as cv

cap = cv.VideoCapture('Image/LAB3_Video.mp4')
if not cap.isOpened():
    print("Error: 비디오 파일을 열 수 없습니다.")
    exit()

kernel = np.ones((3, 5), np.uint8)

while True:
    ret, source = cap.read()
    if not ret:
        break
    source = cv.resize(source, (1920, 1080))

    # =====================================edge===========================
    source_blue, source_green, source_red = cv.split(source)
    v = np.median(source_red)
    lower = int(max(0, 0.90 * v))
    upper = int(min(255, 1.80 * v))
    source_edges = cv.Canny(source_red, lower, upper)

    #==================================roi========================================
    binary = source_edges.copy()
    mask = np.zeros_like(binary)

    pts = np.array([[
        (3, 468),
        (790, 380),
        (495, 1080),
        (3, 1080)
    ]], dtype=np.int32)

    cv.fillPoly(mask, pts, 255)
    roi = cv.bitwise_and(mask, source_edges)
    roi = cv.GaussianBlur(roi, (5, 5), 5)
    roi = cv.morphologyEx(roi, cv.MORPH_CLOSE, kernel)

    #==========================contour===========================
    no_filtered_contours_image = np.zeros_like(binary)
    filtered_contours_image = np.zeros_like(binary)

    contours_roi, _ = cv.findContours(roi, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(no_filtered_contours_image, contours_roi, -1, 255, 2)

    contours_filtered = []
    for cnt in contours_roi:
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        ratio = h / w if w != 0 else 0
        if (area >= 1300) or (600 <= area < 1300 and ratio > 0.7):
            contours_filtered.append(cnt)

    cv.drawContours(filtered_contours_image, contours_filtered, -1, 255, 2)

    #===============================fitting=========================
    if not contours_filtered:
        continue

    all_points = np.vstack(contours_filtered).squeeze()
    all_points = all_points[np.argsort(all_points[:, 0])]
    x = all_points[:, 0]
    y = all_points[:, 1]

    coeffs = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly(x_fit)

    final_line = np.zeros_like(binary)
    final_line = cv.cvtColor(final_line, cv.COLOR_GRAY2BGR)

    # 빈 이미지에 **점선**으로 피팅된 곡선을 그려넣는다.
    for i in range(len(x_fit) - 1):
        pt1 = (int(x_fit[i]), int(y_fit[i]))
        pt2 = (int(x_fit[i + 1]), int(y_fit[i + 1]))
        if 0 <= pt1[1] < source.shape[0] and 0 <= pt2[1] < source.shape[0]:
            if i % 4 == 0:  # 점선을 위해 4번째 선마다 그린다
                cv.line(final_line, pt1, pt2, (0, 255, 0), 2)

    final_output = cv.addWeighted(source, 1.0, final_line, 1.0, 0)

    #===============================score=======================================
    final_line_for_score = final_line.copy()
    final_line_for_score = cv.cvtColor(final_line_for_score, cv.COLOR_BGR2GRAY)
    points_final_line = cv.findNonZero(final_line_for_score)

    bottom_image, width = final_output.shape[:2]
    ys = points_final_line[:, 0, 1]
    score = bottom_image - np.max(ys)

    threshold_1 = int(bottom_image - 250)
    threshold_2 = int(bottom_image - 120)

    if score > 250:
        level = 1
    elif 120 <= score <= 250:
        level = 2
    else:
        level = 3

    # 점선으로 기준선 추가
    for y_val, color in zip([threshold_1, threshold_2], [(0, 255, 0), (255, 0, 0)]):
        for x_pos in range(0, width, 20):
            cv.line(final_output, (x_pos, y_val), (x_pos + 10, y_val), color, 2)

    #===============================Text Box=======================================
    # 텍스트 박스 위치
    box_x, box_y, box_w, box_h = 1400, 350, 250, 100
    cv.rectangle(final_output, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)

    # Score
    cv.putText(final_output,
               f'Score : {score:.2f}',
               (box_x + 10, box_y + 40),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2,
               cv.LINE_AA)

    # Level
    cv.putText(final_output,
               f'Level : {level}',
               (box_x + 10, box_y + 85),
               cv.FONT_HERSHEY_SIMPLEX,
               1,
               (255, 255, 255),
               2,
               cv.LINE_AA)

    # ===========================Display==========================
    display_scale = 0.5

    display_frame = cv.resize(final_output, (int(width * display_scale), int(bottom_image * display_scale)))
    cv.imshow('final output', display_frame)

    display_frame = cv.resize(final_line, (int(width * display_scale), int(bottom_image * display_scale)))
    cv.imshow('final line', display_frame)

    display_frame = cv.resize(no_filtered_contours_image, (int(width * display_scale), int(bottom_image * display_scale)))
    cv.imshow('no_filtered_contours_image', display_frame)

    display_frame = cv.resize(filtered_contours_image, (int(width * display_scale), int(bottom_image * display_scale)))
    cv.imshow('filtered_contours_image', display_frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
