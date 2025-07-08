#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

int main()
{
	//동영상의 한 프레임을 가져와 이미지처리를 하기 위한 행렬
	Mat image;

	//hsv스케일로 이미지처리를 위함
	Mat hsv;

	//inRange함수를 거친 이후 나무판자와 같은 색깔의 부분만 255, 나머지는 0으로 채운 binary 이미지
	Mat mask;

	//dilate를 위한 커널
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

	//휴대폰의 hsv값
	int hmin = 25, hmax = 45, smin = 90, smax = 255, vmin = 112, vmax = 224;


	//주어진 동영상을 가져온다. 
	VideoCapture cap("LAB_MagicCloak_Sample2.mp4");

	


	//동영상을 가져오지 못했을 경우 에러메세지를 띄운다. 
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	//원본 이미지에서 마스크 부분에 뒷 배경이 나오게 하기 위해 맨 처음 프레임을 저장한다. 
	Mat background;
	cap.read(background);


	while (true)
	{

		// 매 프레임을 읽어온다.
		Mat src;
		bool bSuccess = cap.read(src);

		//이미지처리를 위해 소스이미지를 복사하여 가져온다. 
		image = src.clone();

		//최종 결과물을 송출할 이미지. 
		Mat image_disp;

		//이미지를 가져오지 못했을 경우 루프를 종료시킨다. 
		if (!bSuccess)
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}

		// esc키를 30ms간 기다리고, esc키가 눌리면 프로그램을 종료한다. 


		//기존 BGR스케일이던 이미지를 HSV스케일로 바꾼다. 
		cvtColor(image, hsv, COLOR_BGR2HSV);

		//Smoothing하여 이미지의 노이즈를 제거한다. 
		GaussianBlur(hsv, hsv, Size(3, 3), 0);

		//기존에 찾았던 나무판자의 hsv값을 기반으로 mask를 만든다. 
		inRange(hsv, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), mask);

		//마스크 이미지가 깨지는 것을 막기 위해 Morphology를 적용한다. 
		erode(mask, mask, kernel);
		morphologyEx(mask, mask, MORPH_DILATE, kernel, Point(-1, -1), 2);

		
		//BGR채널 이미지와 bitwise연산을 하기 위해 mask의 스케일을 바꿔준다. 
		cvtColor(mask, mask, COLOR_GRAY2BGR);

		//배경이 보이는 나무판자를 넣기 위한 행렬
		Mat cloak;

		//마스크 부분에 배경이 나오도록 처리한다. 
		bitwise_and(background, mask, cloak);
		
		//나무판자부분이 0인 이미지를 넣기 위한 행렬
		Mat outside_cloak;
		//기존 이미지에서 마스크부분을 0으로 처리한다. 
		bitwise_not(mask, mask);

		//마스크 부분만 검정색인 이미지를 만들어낸다. 
		bitwise_and(image, mask, outside_cloak);
		
		//각각의 이미지를 더하여 최종 결과물을 만들어낸다. 
		bitwise_or(outside_cloak, cloak, image_disp);

		//최종 이미지를 출력한다. 
		imshow("image disp", image_disp);
		

		//esc키를 누르거나 영상이 끝나면 동영상을 최종적으로 저장한다. 
		char ch = waitKey(10);
		if (ch == 27)
		{
			cout << "ESC key is pressed by user\n";
	
			break;
		}
	}

	return 0;
}
