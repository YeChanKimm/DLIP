#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;


int main()
{
	//�������� �� �������� ������ �̹���ó���� �ϱ� ���� ���
	Mat image;

	//hsv�����Ϸ� �̹���ó���� ����
	Mat hsv;

	//inRange�Լ��� ��ģ ���� �������ڿ� ���� ������ �κи� 255, �������� 0���� ä�� binary �̹���
	Mat mask;

	//dilate�� ���� Ŀ��
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

	//���������� hsv��
	int hmin = 7, hmax = 28, smin = 90, smax = 158, vmin = 131, vmax = 189;

	//������ ��ȭ������ ����
	bool bRec = false;

	//�־��� �������� �����´�. 
	VideoCapture cap("../LAB_MagicCloak_Sample1.mp4");

	

	//�������� �������� ������ ��� �����޼����� ����. 
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	//���� �̹������� ����ũ �κп� �� ����� ������ �ϱ� ���� �� ó�� �������� �����Ѵ�. 
	Mat background;
	cap.read(background);


	while (true)
	{
		// �� �������� �о�´�.
		Mat src;
		bool bSuccess = cap.read(src);

		//�̹���ó���� ���� �ҽ��̹����� �����Ͽ� �����´�. 
		image = src.clone();

		//���� ������� ������ �̹���. 
		Mat image_disp;

		//����� ���̴� �������ڸ� �ֱ� ���� ���
		Mat cloak;

		//�������ںκ��� 0�� �̹����� �ֱ� ���� ���
		Mat outside_cloak;

		//�̹����� �������� ������ ��� ������ �����Ų��. 
		if (!bSuccess)
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}

		//���� BGR�������̴� �̹����� HSV�����Ϸ� �ٲ۴�. 
		cvtColor(image, hsv, COLOR_BGR2HSV);

		//Smoothing�Ͽ� �̹����� ����� �����Ѵ�. 
		GaussianBlur(hsv, hsv, Size(3, 3), 0);

		//������ ã�Ҵ� ���������� hsv���� ������� mask�� �����. 
		inRange(hsv, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), mask);

		//����ũ �̹����� ������ ���� ���� ���� morphology�� �����Ѵ�. 
		morphologyEx(mask, mask, MORPH_DILATE, kernel, Point(-1, -1), 1);
		

		//BGRä�� �̹����� bitwise������ �ϱ� ���� mask�� �������� �ٲ��ش�. 
		cvtColor(mask, mask, COLOR_GRAY2BGR);

		//����ũ �κп� ����� �������� ó���Ѵ�. 
		bitwise_and(background, mask, cloak);

		//���� �̹������� ����ũ�κ��� 0���� ó���� �� ���� �̹����� ����ũ�� ������ �κ��� �������� ó���Ѵ�. 
		bitwise_not(mask, mask);
		bitwise_and(image, mask, outside_cloak);

		//������ �̹����� ���Ͽ� ���� ������� ������. 
		bitwise_or(outside_cloak, cloak, image_disp);

		//���� �̹����� ����Ѵ�.		
		imshow("image disp", image_disp);
		
		

		//escŰ�� �����ų� ������ ������ �������� ���������� �����Ѵ�. 
		char ch = waitKey(10);
		if (ch == 27)
		{
			cout << "ESC key is pressed by user\n";
			break;
		}
	}

	return 0;
}
