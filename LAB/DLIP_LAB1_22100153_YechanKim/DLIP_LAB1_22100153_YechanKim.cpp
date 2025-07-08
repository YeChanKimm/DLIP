//#include<iostream>
//#include<opencv2/opencv.hpp>
//#include <iomanip>
//
//using namespace std;
//using namespace cv;
//
//// �Լ� ����
//void finding_defective_teeth(Mat& std);
//
//int main()
//{
//
//
//    // �� ��� �̹����� �ҷ��´�.
//    Mat gear1 = imread("Gear1.jpg", IMREAD_GRAYSCALE);
//    Mat gear2 = imread("Gear2.jpg", IMREAD_GRAYSCALE);
//    Mat gear3 = imread("Gear3.jpg", IMREAD_GRAYSCALE);
//    Mat gear4 = imread("Gear4.jpg", IMREAD_GRAYSCALE);
//
//    //�̹����� �ε���� ���� �ÿ� ���� �޼����� ����Ѵ�.
//    if (gear1.empty() || gear2.empty() || gear3.empty() || gear4.empty()) {
//        cerr << "One or more images failed to load. Check the file paths." << endl;
//        return -1;
//    }
//
//
//
//
//    
//
//    /*
//    gear1~4�� ���� �����̻� ������ ��ġ�� ����Ѵ�. 
//   
//   
//    !!��� �̹����� ���� �����̽��ٸ� ���� �� ���� �̹����� ��µȴ�.!!
//
//    */
//    cout << "Gear1" << endl;
//    finding_defective_teeth(gear1);
//
//
//    cout << "Gear2" << endl;
//    finding_defective_teeth(gear2);
//
//
//    cout << "Gear3" << endl;
//    finding_defective_teeth(gear3);
//
//
//    cout << "Gear4" << endl;
//    finding_defective_teeth(gear4);
//
//    return 0;
//}
//
//void finding_defective_teeth(Mat& std)
//{
//    //����̻��� ��ġ�� �������� ǥ���ϰ� ���� �Էµ� �̹����� �÷��� ��ȯ�Ͽ� ���ο� ��Ŀ� �����Ѵ�. 
//    Mat std_colored;
//    cvtColor(std, std_colored, COLOR_GRAY2BGR);
//
//
//    //OpenCV���� �����ϴ� ������ threshold �Ӱ谪�� �����ϴ� �Լ��� ����Ѵ�. 
//    //threshold value: gear1-->7 gear2-->8 gear3-->7 gear4-->8
//    Mat after_threshold;
//    double otsu_thresh_val = threshold(std, after_threshold, 0, 255, THRESH_BINARY | cv::THRESH_OTSU);
//
//    //�̻��� ������ �����Ѵ�. 
//    int teeth_number = 20;
//
//    //�������� ���ϱ� ���� ��� ���� �ٱ����� ����� ã�´�. 
//    vector<vector<Point>> contours_whole;
//    findContours(after_threshold, contours_whole, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//    //�������� �Ķ���ͷ� ������ �������� �����Ѵ�.
//    Point2f center;
//    float radius;
//
//    //2D �̹������� ���� �ٱ��ʿ� �ִ� �κ��� �������� �Ͽ� �������� ���Ѵ�. 
//    int maxIndex = 0;
//    double maxArea = 0;
//    for (int i = 0; i < contours_whole.size(); i++) {
//        double area = contourArea(contours_whole[i]);
//        if (area > maxArea) {
//            maxArea = area;
//            maxIndex = i;
//        }
//    }
//    minEnclosingCircle(contours_whole[maxIndex], center, radius);
//
//
//
//    //��� ��ü���� �̻��� ����� ���� �̻��� ���̸� ���Ѵ�. (�̻�����=���*2.25)
//    double module = (2 * radius) / (teeth_number + 2);
//    double teeth_length = module * 2.25;
//    float inner_radius = radius - teeth_length;
//
//
//    //root diameter�� ���Ѵ�.
//    double root_diameter = module * teeth_number - 2 * teeth_length + 2 * module;
//
//
//    //�� �̹������� ���ο��� �� �̻��� �����. 
//    Mat teeth = after_threshold.clone();
//    circle(teeth, center, inner_radius, Scalar(0), -1);
//
//    //�̻��� ���� ����Ǿ��ִ� ���� ����� ���� �� ����� ��Ȯ������ �ʱ⿡ erode�� �����Ѵ�. 
//    Mat teeth_morphology;
//    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//    erode(teeth, teeth_morphology, kernel);
//
//    // �̻��� ���� ����� ���Ѵ�. 
//    vector<vector<Point>> contours_teeth;
//    findContours(teeth_morphology, contours_teeth, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//
//
//    //�̻� ������ ����� ���Ѵ�. 
//    float teeth_area_sum = 0;
//    for (const auto& contour : contours_teeth) {
//        teeth_area_sum += contourArea(contour);
//    }
//    float teeth_area_average = teeth_area_sum / teeth_number;
//
//    //�̻��� ����� �׷����� ����� �����Ѵ�. 
//    Mat drawing_with_text = Mat::zeros(std.size(), CV_8UC3);
//    Mat drawing = Mat::zeros(std.size(), CV_8UC3);
//
//    //�����̻��� ������ ���� ���� ������ �����Ѵ�. 
//    int defective_teeth_count = 0;
//
//    //�̻� �ϳ��ϳ��� ���� ���԰˻縦 �Ѵ�. 
//    for (int i = 0; i < contours_teeth.size(); i++) {
//        double teeth_area = contourArea(contours_teeth[i]);
//
//        //���� Ư�� �̻��� ������ ���-200���� �۰ų�, ���+200���� Ŭ ��� ������ �ִٰ� �Ǵ��Ѵ�. 
//        if ((0 < teeth_area && teeth_area < (teeth_area_average - 200)) || (teeth_area > (teeth_area_average + 200))) {
//            
//            // �ؽ�Ʈ�� ���� �̹����� �� �̹��� ��� ���� �̻��� ���������� ����� �׸���. 
//            drawContours(drawing, contours_teeth, i, Scalar(0, 0, 255), 2);
//            drawContours(drawing_with_text, contours_teeth, i, Scalar(0, 0, 255), 2);
//
//            // �ؽ�Ʈ�� ������ ǥ���Ѵ�. 
//            putText(drawing_with_text, to_string((int)teeth_area), contours_teeth[i][0], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
//
//            // ��� �������� �÷� �̹��� ���� ����̻��� ��ġ�� ǥ���Ѵ�. 
//            Moments M = moments(contours_teeth[i]);
//            int cx = int(M.m10 / M.m00);
//            int cy = int(M.m01 / M.m00);
//            Point center_teeth(cx, cy);
//            for (int angle = 0; angle < 360; angle += 20) {
//                ellipse(std_colored, center_teeth, Size(30, 30), 0, angle, angle + 10, Scalar(0, 255, 255), 2);
//            }
//
//            //�����̻��� ������ �߰��Ѵ�. 
//            defective_teeth_count++;
//        }
//
//        //���-200���� ũ��, ���+200���� ���� �̻��� �����̻��� ó���Ѵ�. 
//        else if ((teeth_area > (teeth_area_average - 200)) && (teeth_area < (teeth_area_average + 200))) {
//            
//            // �ؽ�Ʈ�� ���� �̹����� �� �̹��� ��� ���� �̻��� �ʷϻ����� ����� �׸���. 
//            drawContours(drawing, contours_teeth, i, Scalar(0, 255, 0), 2);
//            drawContours(drawing_with_text, contours_teeth, i, Scalar(0, 255, 0), 2);
//
//            //�̻��� ������ �����Ѵ�. 
//            putText(drawing_with_text, to_string((int)teeth_area), contours_teeth[i][0], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
//        }
//    }
//
//
//
//    //��պ����� ǥ�õ� �÷��̹����� ǥ���Ѵ�. 
//    namedWindow("gear", WINDOW_AUTOSIZE);
//    imshow("gear", std_colored);
//
//
//    //����̻��� �Ϲ� �̻��� ������ ������ ������ �̹����� ǥ���Ѵ�. 
//    namedWindow("contours with text", WINDOW_AUTOSIZE);
//    imshow("contours with text", drawing_with_text);
//
//    namedWindow("contours", WINDOW_AUTOSIZE);
//    imshow("contours", drawing);
//
//
//
//    //�̻� ����, ���, ����̻� ����, ����� root diameter�� ����Ѵ�. 
//    cout << "Teeth numbers: " << teeth_number << endl;
//    cout << "Avg. Teeth Area: " << teeth_area_average << endl;  
//    cout << "Defective Teeth: " << defective_teeth_count << endl;
//    cout << "Diameter of the gear:" << root_diameter << endl;
//  
//    // ����Ƽ �˻縦 ����Ѵ�. 
//    if (defective_teeth_count > 0)
//        cout << "Quality : Fail" << endl;
//    else
//        cout << "Quality : Pass" << endl;
//    cout << endl;
//
//
//    waitKey(0);
//}
//
//
