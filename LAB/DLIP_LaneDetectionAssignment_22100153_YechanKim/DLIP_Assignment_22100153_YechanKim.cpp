#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

Point getIntersection(Vec4i l1, Vec4i l2);
void drawExtendedLine(Mat& img, Vec4i line, Scalar color, int thickness, int height);
void lane_detecting(Mat& src, const string& window_name);


int main()
{
	//���� �̹����� �����´�. 
	Mat Lane_center = imread("Lane_center.jpg", IMREAD_COLOR);
	Mat Lane_changing = imread("Lane_changing.jpg", IMREAD_COLOR);

	//�̹����� �ҷ����� �������� ��� ���� �޼����� ������. 
	if (Lane_center.empty() || Lane_changing.empty()) {
		cerr << "�̹����� �ҷ����� �� �����߽��ϴ�." << endl;
		return -1;
	}

	//�� ���� �̹����� ���ؼ� ������ �����Ѵ�. 
	lane_detecting(Lane_center, "Center View");
	lane_detecting(Lane_changing, "Changing View");


	waitKey(0);
	return 0;
}

// �� ���� ���������� �˷��ִ� �Լ�
Point getIntersection(Vec4i l1, Vec4i l2) {
	Point2f a1(l1[0], l1[1]), a2(l1[2], l1[3]); // �� 1�� �� ��
	Point2f b1(l2[0], l2[1]), b2(l2[2], l2[3]); // �� 2�� �� ��

	float d = (a1.x - a2.x) * (b1.y - b2.y) - (a1.y - a2.y) * (b1.x - b2.x);
	if (d == 0) return Point(-1, -1); // ����: ������ ����

	//ũ���� ���� ���
	float px = ((a1.x * a2.y - a1.y * a2.x) * (b1.x - b2.x) - (a1.x - a2.x) * (b1.x * b2.y - b1.y * b2.x)) / d;
	float py = ((a1.x * a2.y - a1.y * a2.x) * (b1.y - b2.y) - (a1.y - a2.y) * (b1.x * b2.y - b1.y * b2.x)) / d;

	return Point(cvRound(px), cvRound(py));
}



//ª�� ���� ���� ������ �÷��ִ� �Լ�
void drawExtendedLine(Mat& img, Vec4i line, Scalar color, int thickness, int height) {
	double x1 = line[0], y1 = line[1];
	double x2 = line[2], y2 = line[3];

	double dx = x2 - x1;
	double dy = y2 - y1;

	// slope ������ üũ
	if (abs(dx) < 1e-3) return;

	double slope = dy / dx;
	double intercept = y1 - slope * x1;

	// y = slope * x + intercept
	int y_top = int(height * 0.6);
	int y_bottom = height;

	int x_top = int((y_top - intercept) / slope);
	int x_bottom = int((y_bottom - intercept) / slope);

	// ��ȿ ��ǥ �˻�
	if (x_top < 0 || x_top > img.cols || x_bottom < 0 || x_bottom > img.cols)
		return;

	cv::line(img, Point(x_bottom, y_bottom), Point(x_top, y_top), color, thickness);
}



void lane_detecting(Mat& src, const string& window_name)
{
	Mat src_gray;

	//������ �̹����� �׷��̽����Ϸ� �ٲ۴�. 
	cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);



	int width = src.cols;
	int height = src.rows;

	//���� �ֺ� ��ٸ��� ������� roi�� ����. 
	Point pts[4] = {
		Point(width * 0.1, height),              // ���ϴ�
		Point(width * 0.45, height * 0.6),       // �»��
		Point(width * 0.55, height * 0.6),       // ����
		Point(width * 0.95, height)               // ���ϴ�
	};
	Mat mask = Mat::zeros(src_gray.size(), src_gray.type());
	fillConvexPoly(mask, pts, 4, Scalar(255));
	Mat roi_image;
	bitwise_and(src_gray, mask, roi_image);


	//roi �̹����� ���� ������ ���͸� �����Ѵ�. 
	blur(roi_image, roi_image, Size(3, 3));
	

	//���������͸� ������ roi�̹����� ���� canny edge detecting ����� ����Ͽ� ������ �����Ѵ�. 
	Mat edge;
	Canny(roi_image, edge, 10, 100, 3);



	//�������� ������ �� linesP�� ������ �����Ѵ�. 
	vector<Vec4i> linesP;
	HoughLinesP(edge, linesP, 1, CV_PI / 180, 20, 20, 15);
	


	//�߾����κ��� ��������� �������ִ����� ������ ��ȣ�� �������� ���� ������ ���� �и��Ѵ�. 
	vector<Vec4i> left_lines, right_lines;

	for (auto& l : linesP) {
		double slope = double(l[3] - l[1]) / (l[2] - l[0] + 1e-6);
		int x_center = (l[0] + l[2]) / 2;

		if (slope < -0.3 && x_center < width / 2)
			left_lines.push_back(l);
		else if (slope > 0.3 && x_center > width / 2)
			right_lines.push_back(l);
	}


	//���� ���� �� �� ������ �����Ͽ� ������ ������ �����ϰ� �ʱ�ȭ�Ѵ�. 
	Vec4i left_best=Vec4i(0, 0, 0, 0);
	Vec4i right_best = Vec4i(0, 0, 0, 0); 

	//���� ��, ������ �� ��� �߾Ӱ� ���� ����� ���� �������� �����Ͽ� left_best, right_best������ �Է��Ѵ�. 
	int max_x_center = 0;
	for (auto& l : left_lines) {
		int x_center = (l[0] + l[2]) / 2;
		if (x_center > max_x_center) {
			max_x_center = x_center;
			left_best = l;
		}
	}

	int min_x_center = width; 
	for (auto& l : right_lines) {
		int x_center = (l[0] + l[2]) / 2;
		if (x_center < min_x_center) {
			min_x_center = x_center;
			right_best = l;
		}
	}

	//���� ������� �� �̹����� �����. 
	Mat output = src.clone();

	//���� ���� ������ ���� ������ ������ �ҽ������� �Ͽ� �̹����� �ʷϻ� ������ �׸���.
	Point vanish = getIntersection(left_best, right_best); 
	circle(output, vanish, 10, Scalar(0, 0, 255), 5);  

	//�ü������� ���� �Ķ������� �׸���.  
	line(output, vanish, Point(vanish.x, height), Scalar(255, 0, 0), 1);

	//���������� ������ ���� ������ ������ ������ �����Ͽ� �ð������� ������ ǥ���Ѵ�. 
	drawExtendedLine(output, left_best, Scalar(0, 0, 255), 2, height);      
	drawExtendedLine(output, right_best, Scalar(0, 255, 0), 2, height); 

	//�������ϰ� ���� �� ���θ� ǥ���ϱ� ���� ����� �����Ѵ�. 
	Mat overlay = output.clone();

	// ����� ���� ���� ������ ���� �Ʒ���, ���� ���� ����Ѵ�. 
	// y_top�� drawExtendedLine() �������� 0.6*height
	int y_top = int(height * 0.6);
	int y_bottom = height;

	// ���� ���� ���Ʒ� ��
	double lx1 = left_best[0], ly1 = left_best[1];
	double lx2 = left_best[2], ly2 = left_best[3];
	double lslope = (ly2 - ly1) / (lx2 - lx1 + 1e-6);
	double interceptL = ly1 - lslope * lx1;
	int lxtop = int((y_top - interceptL) / lslope);
	int lxbottom = int((y_bottom - interceptL) / lslope);

	// ������ ���� ���Ʒ� ��
	double rx1 = right_best[0], ry1 = right_best[1];
	double rx2 = right_best[2], ry2 = right_best[3];
	double rslope = (ry2 - ry1) / (rx2 - rx1 + 1e-6);
	double interceptR = ry1 - rslope * rx1;
	int rxtop = int((y_top - interceptR) / rslope);
	int rxbottom = int((y_bottom - interceptR) / rslope);

	// ���� ������ �����ϴ� ��ٸ��� ��ǥ 
	vector<Point> road_area = {
		Point(lxbottom, y_bottom),
		Point(lxtop, y_top),
		Point(rxtop, y_top),
		Point(rxbottom, y_bottom)
	};

	// ������ ä���. (�����)
	fillConvexPoly(overlay, road_area, Scalar(255, 0, 255));

	// �������ϰ� �ռ��Ѵ�.  (alpha blending)
	double alpha = 0.2;
	addWeighted(overlay, alpha, output, 1 - alpha, 0, output);

	//���� ������� ����Ѵ�. 

	namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	imshow(window_name, output);


}