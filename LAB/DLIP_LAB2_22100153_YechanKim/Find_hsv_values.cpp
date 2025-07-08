//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//using namespace std;
//
////루프 속에서 동영상의 프레임을 받아올 변수
//Mat src;
//
////selection을 만들 때 기존 좌표를 표시하기 위함
//Point origin;
//
////ROI가 지정될 영역
//Rect selection;
//
////마우스로 특정 영역이 지정되었는지 확인하기 위함.
//bool selectObject = false;
//bool trackObject = false;
//
//
////hsv스케일의 전체 범위
//int hmin = 0, hmax = 179, smin = 0, smax = 255, vmin = 0, vmax = 255;
//
//// On mouse 이벤트 
//static void onMouse(int event, int x, int y, int, void*);
//
//int main()
//{
//	//최종적으로 화면에 보일 이미지
//	Mat image_disp;
//	
//	//hsv스케일로 바꾼 이미지를 저장할 공간
//	Mat hsv;
//
//	//inRang의 결과물이 저장될 공간
//	Mat dst;
//	
//	//샘플 동영상
//	VideoCapture cap("../LAB_MagicCloak_Sample2.mp4");
//	//VideoCapture cap("../LAB_MagicCloak_Sample1.mp4");
//
//	//동영상을 불러오지 못했다면 에러 메세지를 출력한다. 
//	if (!cap.isOpened())	
//	{
//		cout << "Cannot open the video cam\n";
//		return -1;
//	}
//
//
//	 //설정한 색깔의 hsv정보를 나타낼 트랙바 설정
//	namedWindow("Source", 0);
//	
//	//콜백함수를 onMouse로 설정
//	setMouseCallback("Source", onMouse, 0);
//
//	//트랙바 설정
//	createTrackbar("Hmin", "Source", &hmin, 179, 0);
//	createTrackbar("Hmax", "Source", &hmax, 179, 0);
//	createTrackbar("Smin", "Source", &smin, 255, 0);
//	createTrackbar("Smax", "Source", &smax, 255, 0);
//	createTrackbar("Vmin", "Source", &vmin, 255, 0);
//	createTrackbar("Vmax", "Source", &vmax, 255, 0);
//
//
//	//waitKey함수를 위한 변수 설정
//	int key = 0;
//	
//	while (true)
//	{
//		//동영상을 한 프레임씩 받아와 src에 저장한다. 
//		bool bSuccess = cap.read(src);
//
//		if (!bSuccess) {
//			cout << "End of video stream or error.\n";
//			destroyAllWindows();  // 창 닫기
//			break;
//		}
//		
//		//30ms간 기다리고, esc가 눌리면 루프를 빠져나간다.
//		key = waitKey(30);
//		if (key == 27) 
//		{
//			cout << "ESC key is pressed by user\n";
//			break;
//		}
//	
//
//		//받아온 이미지를 hsv스케일로 변환시킨다. 
//		cvtColor(src, hsv, COLOR_BGR2HSV); 
//
//		//노이즈 처리를 위해 smoothing한다. 
//		GaussianBlur(hsv, hsv, Size(3, 3),0);
//
//		
//		//영역이 선택되기 이전 hsv값을 초기화하여 dst에 저장한다. 
//		inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
//			Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), dst);// 보고자 하는 색상정보
//
//		imshow("dst", dst);
//		//마우스로 선택된 영역이 0보다 클 경우 if문이 작동한다. 
//		if (trackObject)
//		{
//			//해당 이벤트가 계속하여 발생하지 않도록 초기화한다. 
//			trackObject = false;
//			
//			//hsv스케일로 선택된 영역에 대해 roi영역을 만들기 위한 변수
//			Mat roi_HSV(hsv, selection); 	
//
//			//선택된 영역의 각 h,s,v에 대해 평균과 표준편차를 구한다. 
//			Scalar means, stddev;
//			meanStdDev(roi_HSV, means, stddev);
//			cout << "\n Selected ROI Means= " << means << " \n stddev= " << stddev;
//
//
//			//트랙바의 값을 평균과 표준편차를 이용하여 변환시킨다. (평균-표준편차=최솟값, 평균+표준편차=최댓값) 
//			hmin = MAX((means[0] - stddev[0]), 0);
//			hmax = MIN((means[0] + stddev[0]), 179);
//			setTrackbarPos("Hmin", "Source", hmin);
//			setTrackbarPos("Hmax", "Source", hmax);
//
//			smin = MAX((means[1] - stddev[1]), 0);
//			smax = MIN((means[1] + stddev[1]), 255);
//			setTrackbarPos("Smin", "Source", smin);
//			setTrackbarPos("Smax", "Source", smax);
//
//			vmin = MAX((means[2] - stddev[2]), 0);
//			vmax = MIN((means[2] + stddev[2]), 255);
//			setTrackbarPos("Vmin", "Source", vmin);
//			setTrackbarPos("Vmax", "Source", vmax);
//
//		}
//		src.copyTo(image_disp);
//
//		if (selectObject && selection.area() > 0)  // Left Mouse is being clicked and dragged
//		{
//			 //Mouse Drag을 화면에 보여주기 위함
//			Mat roi_RGB(image_disp, selection);
//			bitwise_not(roi_RGB, roi_RGB);
//
//		}
//		
//		//최종적으로 처리된 이미지를 출력한다. 		
//		
//		imshow("Source", image_disp);
//
//	} 
//
//	return 0;
//}
//
//
//
// //On mouse 이벤트 
//static void onMouse(int event, int x, int y, int, void*)
//{
//
//	if (src.empty()) return;
//
//	//선택된 영역을 roi로 설정한다. 
//	if (selectObject && !src.empty())
//	{
//		//드레그를 시작한 순간과 이후 좌표를 비교하여 selection의 상단 좌표를 설정한다. 
//		selection.x = MIN(x, origin.x);
//		selection.y = MIN(y, origin.y);
//
//		//roi의 높이와 넓이를 설정한다. 
//		selection.width = abs(x - origin.x) + 1;
//		selection.height = abs(y - origin.y) + 1;
//
//		//roi를 적용시킨다. 
//		selection &= Rect(0, 0, src.cols, src.rows);
//	}
//
//	//마우스 버튼을 누르고 일정 영역을 선택하면 selectObject가 참이 된다. 
//	switch (event)
//	{
//	case EVENT_LBUTTONDOWN:
//		selectObject = true;
//		origin = Point(x, y);
//		break;
//	case EVENT_LBUTTONUP:
//		selectObject = false;
//		if (selection.area())
//			trackObject = true;
//		break;
//	}
//}