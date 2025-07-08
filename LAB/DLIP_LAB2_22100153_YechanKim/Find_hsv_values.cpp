//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//using namespace std;
//
////���� �ӿ��� �������� �������� �޾ƿ� ����
//Mat src;
//
////selection�� ���� �� ���� ��ǥ�� ǥ���ϱ� ����
//Point origin;
//
////ROI�� ������ ����
//Rect selection;
//
////���콺�� Ư�� ������ �����Ǿ����� Ȯ���ϱ� ����.
//bool selectObject = false;
//bool trackObject = false;
//
//
////hsv�������� ��ü ����
//int hmin = 0, hmax = 179, smin = 0, smax = 255, vmin = 0, vmax = 255;
//
//// On mouse �̺�Ʈ 
//static void onMouse(int event, int x, int y, int, void*);
//
//int main()
//{
//	//���������� ȭ�鿡 ���� �̹���
//	Mat image_disp;
//	
//	//hsv�����Ϸ� �ٲ� �̹����� ������ ����
//	Mat hsv;
//
//	//inRang�� ������� ����� ����
//	Mat dst;
//	
//	//���� ������
//	VideoCapture cap("../LAB_MagicCloak_Sample2.mp4");
//	//VideoCapture cap("../LAB_MagicCloak_Sample1.mp4");
//
//	//�������� �ҷ����� ���ߴٸ� ���� �޼����� ����Ѵ�. 
//	if (!cap.isOpened())	
//	{
//		cout << "Cannot open the video cam\n";
//		return -1;
//	}
//
//
//	 //������ ������ hsv������ ��Ÿ�� Ʈ���� ����
//	namedWindow("Source", 0);
//	
//	//�ݹ��Լ��� onMouse�� ����
//	setMouseCallback("Source", onMouse, 0);
//
//	//Ʈ���� ����
//	createTrackbar("Hmin", "Source", &hmin, 179, 0);
//	createTrackbar("Hmax", "Source", &hmax, 179, 0);
//	createTrackbar("Smin", "Source", &smin, 255, 0);
//	createTrackbar("Smax", "Source", &smax, 255, 0);
//	createTrackbar("Vmin", "Source", &vmin, 255, 0);
//	createTrackbar("Vmax", "Source", &vmax, 255, 0);
//
//
//	//waitKey�Լ��� ���� ���� ����
//	int key = 0;
//	
//	while (true)
//	{
//		//�������� �� �����Ӿ� �޾ƿ� src�� �����Ѵ�. 
//		bool bSuccess = cap.read(src);
//
//		if (!bSuccess) {
//			cout << "End of video stream or error.\n";
//			destroyAllWindows();  // â �ݱ�
//			break;
//		}
//		
//		//30ms�� ��ٸ���, esc�� ������ ������ ����������.
//		key = waitKey(30);
//		if (key == 27) 
//		{
//			cout << "ESC key is pressed by user\n";
//			break;
//		}
//	
//
//		//�޾ƿ� �̹����� hsv�����Ϸ� ��ȯ��Ų��. 
//		cvtColor(src, hsv, COLOR_BGR2HSV); 
//
//		//������ ó���� ���� smoothing�Ѵ�. 
//		GaussianBlur(hsv, hsv, Size(3, 3),0);
//
//		
//		//������ ���õǱ� ���� hsv���� �ʱ�ȭ�Ͽ� dst�� �����Ѵ�. 
//		inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
//			Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), dst);// ������ �ϴ� ��������
//
//		imshow("dst", dst);
//		//���콺�� ���õ� ������ 0���� Ŭ ��� if���� �۵��Ѵ�. 
//		if (trackObject)
//		{
//			//�ش� �̺�Ʈ�� ����Ͽ� �߻����� �ʵ��� �ʱ�ȭ�Ѵ�. 
//			trackObject = false;
//			
//			//hsv�����Ϸ� ���õ� ������ ���� roi������ ����� ���� ����
//			Mat roi_HSV(hsv, selection); 	
//
//			//���õ� ������ �� h,s,v�� ���� ��հ� ǥ�������� ���Ѵ�. 
//			Scalar means, stddev;
//			meanStdDev(roi_HSV, means, stddev);
//			cout << "\n Selected ROI Means= " << means << " \n stddev= " << stddev;
//
//
//			//Ʈ������ ���� ��հ� ǥ�������� �̿��Ͽ� ��ȯ��Ų��. (���-ǥ������=�ּڰ�, ���+ǥ������=�ִ�) 
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
//			 //Mouse Drag�� ȭ�鿡 �����ֱ� ����
//			Mat roi_RGB(image_disp, selection);
//			bitwise_not(roi_RGB, roi_RGB);
//
//		}
//		
//		//���������� ó���� �̹����� ����Ѵ�. 		
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
// //On mouse �̺�Ʈ 
//static void onMouse(int event, int x, int y, int, void*)
//{
//
//	if (src.empty()) return;
//
//	//���õ� ������ roi�� �����Ѵ�. 
//	if (selectObject && !src.empty())
//	{
//		//�巹�׸� ������ ������ ���� ��ǥ�� ���Ͽ� selection�� ��� ��ǥ�� �����Ѵ�. 
//		selection.x = MIN(x, origin.x);
//		selection.y = MIN(y, origin.y);
//
//		//roi�� ���̿� ���̸� �����Ѵ�. 
//		selection.width = abs(x - origin.x) + 1;
//		selection.height = abs(y - origin.y) + 1;
//
//		//roi�� �����Ų��. 
//		selection &= Rect(0, 0, src.cols, src.rows);
//	}
//
//	//���콺 ��ư�� ������ ���� ������ �����ϸ� selectObject�� ���� �ȴ�. 
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