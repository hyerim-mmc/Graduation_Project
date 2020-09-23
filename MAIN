#include <stdio.h>
#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#define OPENCV

#include "yolo_v2_class.hpp"	// imported functions from DLL

#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#include "opencv2/core/version.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#pragma comment(lib, "opencv_world341.lib") 
#else
#pragma comment(lib, "opencv_core2413.lib")  
#pragma comment(lib, "opencv_imgproc2413.lib")  
#pragma comment(lib, "opencv_highgui2413.lib") 
#endif

const int dir[4][2] = { {-1, 0}, {0, 1}, {1, 0}, {0, -1} };

int WIDTH;
int HEIGHT;

const int PERSON = 0;		const int BICYCLE = 1;		const int CAR = 2;
const int MOTORBIKE = 3;	const int BUS = 5;			const int TRUCK = 7;
const int BIRD = 14;		const int CAT = 15;			const int DOG = 16;
const std::vector<int> Objects = { PERSON, BICYCLE, CAR, MOTORBIKE, BUS, TRUCK, BIRD, CAT, DOG };


// 아래 Parameter들 만들어놨으니 수정해서 사용하면 됨

		/* Pre-setting for detecting */
//////////////////////////////////////////////////
												//
const std::string	CFG = "yolov3.cfg";			//
const std::string	WEIGHTS = "yolov3.weights";	//
const std::string	NAMES = "coco.names";		//
const std::string	VIDEO = "video.avi";		//
const int			FILENUM = 1898;				//
												//
//////////////////////////////////////////////////
	/* Must be installed with *.cpp paths */

std::vector<cv::Scalar> Colors;
std::vector<bbox_t> PosInfo(100);
int frameCnt;

/* Parameters to be adjusted */
//////////////////////////////////////////////////
												//
const int totalColor = 16;						//
const double minProb = 0.95;					//
const int maxDistance = 100;					//
												//
//////////////////////////////////////////////////


// Calculate Euclidean distance between two points
double euclideanDist(cv::Point2f& a, cv::Point2f& b) {
	cv::Point2f diff = a - b;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

// Alert warning when the object is approaching
void alert_warning(cv::Mat &mat_img, bbox_t &obj, cv::Scalar color) {
	unsigned int id = obj.track_id;
	unsigned int currFrame = frameCnt;
	cv::Point2f currPos(obj.x + obj.w / 2, obj.y + obj.h / 2);

	bbox_t prevObj = PosInfo[id % 100];
	cv::Point2f prevPos(prevObj.x + prevObj.w / 2, prevObj.y + prevObj.h / 2);
	int prevFrame = prevObj.frames_counter;

	if (prevObj.track_id > 0) {
		// When Distance is smaller than maxDistance, and frame difference is smaller than 30
		if (euclideanDist(prevPos, currPos) < maxDistance && prevFrame + 30 > currFrame) {

			bool left = currPos.x < WIDTH / 2 && currPos.x - prevPos.x > 10;
			bool right = currPos.x > WIDTH / 2 && prevPos.x - currPos.x > 10;
			bool bigger = obj.w - prevObj.w >= 2 && obj.h - prevObj.h >= 2;

			if (left || right || bigger) {
				int bbox_w = (obj.x + obj.w < WIDTH) ? obj.w : WIDTH - obj.x;
				int bbox_h = (obj.y + obj.h < HEIGHT) ? obj.h : HEIGHT - obj.y;

				// Feel color in the bounding box
				cv::Mat roi = mat_img(cv::Rect(obj.x, obj.y, bbox_w, bbox_h));
				cv::Mat paint(roi.size(), CV_8UC3, color);
				double alpha = 0.3;
				cv::addWeighted(paint, alpha, roi, 1.0 - alpha, 0.0, roi);

				if (left && bigger) cv::putText(mat_img, ">> !", currPos, cv::FONT_HERSHEY_COMPLEX, 5, color, 5);
				if (right && bigger) cv::putText(mat_img, "! <<", currPos, cv::FONT_HERSHEY_COMPLEX, 5, color, 5);
			}
		}
	}

	PosInfo[obj.track_id % 100] = obj;
	PosInfo[obj.track_id % 100].frames_counter = currFrame;
}

// Calculate distance using BFS --- 진행중이니 건드리지 X
double calc_distance(std::vector<std::vector<double>>& totalDistance, std::vector<std::vector<bool>>& check, int x, int y) {
	std::queue<std::pair<int, int>> q;
	q.push({ x, y });
	check[x][y] = true;
	double sum = totalDistance[x][y];
	int cnt = 1;

	while (!q.empty()) {
		int r = q.front().first;
		int c = q.front().second;
		double d = totalDistance[r][c];

		for (int i = 0; i < 4; i++) {
			int nr = r + dir[i][0];
			int nc = c + dir[i][1];
			double nd = totalDistance[nr][nc];

			if (0 <= nr && nr < totalDistance.size() && 0 <= nc && nc <= totalDistance[nr].size()) {
				if (check[nr][nc] == false && abs(d - nd) < 2.0) {
					check[nr][nc] = true;
					sum += nd;
					cnt++;
				}
			}
		}
		q.pop();
	}
	return sum / cnt;
}

// Merged
void draw_boxes(cv::Mat& mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, std::vector<cv::Point2f>& projectedPoints, std::vector<double>& distance_vec, unsigned int wait_msec = 0) {
	for (auto &v : result_vec) {
		bool condition = std::find(Objects.begin(), Objects.end(), v.obj_id) != Objects.end();
		cv::Scalar color = Colors[v.track_id % totalColor];
		if (condition) {
			cv::rectangle(mat_img, cv::Rect(v.x, v.y, v.w, v.h), color, 3);
			if (obj_names.size() > v.obj_id)
				putText(mat_img, obj_names[v.obj_id], cv::Point2f(v.x, v.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
			if (v.track_id > 0) { // Only tracked when greater than 0
				putText(mat_img, std::to_string(v.track_id), cv::Point2f(v.x + 5, v.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
				alert_warning(mat_img, v, color);				
			}


			/* !--- 알고리즘 최적화 구현 중입니다 건드리지 마세용 ---! */

			// std::vector<std::vector<double>> totalDistance;		// 2-dimension vector for distance
			// std::vector<std::vector<bool>> check;
			// int size = 0;

			std::vector<double> totalDistance;
			float min_x = v.x + v.w / 4, max_x = v.x + v.w / 4 * 3;
			float min_y = v.y + v.h / 4, max_y = v.y + v.h / 4 * 3;

			for (int i = 0; i < projectedPoints.size(); i++) {
				auto& p = projectedPoints[i];
				if (min_x < p.x && p.x < max_x && min_y < p.y && p.y < max_y) {
					cv::circle(mat_img, p, 3, CV_RGB(255, 0, 0));
					totalDistance.push_back(distance_vec[i]);
				}
				/*
				std::vector<double> tmp;
				std::vector<bool> tmp_ck;
				if (v.x <= p.x && p.x <= v.x + v.w && v.y <= p.y && p.y <= v.y + v.h) {
					cv::circle(mat_img, p, 3, CV_RGB(255, 0, 0));
					while (v.x <= projectedPoints[i].x && projectedPoints[i].x <= v.x + v.w) {
						tmp.push_back(distance_vec[i]);
						tmp_ck.push_back(false);
						i++; size++;
					}
					totalDistance.push_back(tmp);
					check.push_back(tmp_ck);
				}
				i--; size--;
				*/
			}

			std::sort(totalDistance.begin(), totalDistance.end());
			double sum = 0;
			int count = 0;
			for (int i = (int)(totalDistance.size() * 0.2); i < (int)(totalDistance.size() * 0.6); i++) {
				sum += totalDistance[i];
				count++;
			}
			if (5 <= count) {
				double distance = sum / count;
				putText(mat_img, std::to_string(distance), cv::Point2f(v.x, v.y + v.h + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255));
			}

			/*
			if (5 <= size) {
				int mid_row = totalDistance.size() / 2;
				int mid_col = totalDistance[mid_row].size() / 2;
				double distance = calc_distance(totalDistance, check, mid_row, mid_col);

				putText(mat_img, std::to_string(distance), cv::Point2f(v.x, v.y + v.h + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
			}
			*/
		}
	}
	// cv::imshow("window name", mat_img);
	// cv::waitKey(wait_msec);
}

/* Original
void draw_boxes(cv::Mat& mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, unsigned int wait_msec = 0) {
	for (auto &i : result_vec) {
		bool condition = std::find(Objects.begin(), Objects.end(), i.obj_id) != Objects.end();
		cv::Scalar color = Colors[i.track_id % totalColor];
		if (condition) {
			cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
			if (obj_names.size() > i.obj_id)
				putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
			if (i.track_id > 0) { // Only tracked when greater than 0
				putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x + 5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
				alert_warning(mat_img, i, color);
			}
		}
	}
	// cv::imshow("window name", mat_img);
	// cv::waitKey(wait_msec);
}
*/
#endif	// OPENCV

/* Original
void calc_distance(cv::Mat& mat_img, std::vector<bbox_t>& result_vec, std::vector<cv::Point2f>& projectedPoints, std::vector<double>& distance_vec) {
	for (auto &v : result_vec) {
		std::vector<double> totalDistance;
		float min_x = v.x + v.w / 4, max_x = v.x + v.w / 4 * 3;
		float min_y = v.y + v.h / 4, max_y = v.y + v.h / 4 * 3;
		for (int i = 0; i < projectedPoints.size(); i++) {
			cv::Point2f& p = projectedPoints[i];
			if (min_x < p.x && p.x < max_x && min_y < p.y && p.y < max_y) {
				cv::circle(mat_img, p, 3, CV_RGB(255, 0, 0));
				totalDistance.push_back(distance_vec[i]);
			}
		}
		std::sort(totalDistance.begin(), totalDistance.end());
		double sum = 0;
		int count = 0;
		for (int i = (int)(totalDistance.size() * 0.2); i < (int)(totalDistance.size() * 0.8); i++) {
			sum += totalDistance[i];
			count++;
		}
		if (count > 0) {
			double distance = sum / count;
			putText(mat_img, std::to_string(distance), cv::Point2f(v.x, v.y + v.h + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255));
		}
	}
}

*/


void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
	for (auto &i : result_vec) {

		bool condition = std::find(Objects.begin(), Objects.end(), i.obj_id) != Objects.end();
		if (condition && obj_names.size() > i.obj_id) {
			std::cout << obj_names[i.obj_id] << " - ";
			std::cout << "x = " << i.x << ", y = " << i.y
				<< ", w = " << i.w << ", h = " << i.h
				<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
		}
	}
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; file >> line;) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}


// Fill the vector with random colors
void setRandomColors(std::vector<cv::Scalar> &colors, int numColors) {
	cv::RNG rng(0);
	for (int i = 0; i < numColors; i++)
		colors.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
}


int main()
{
	/* Initialization */
	Detector detector(CFG, WEIGHTS);
	auto obj_names = objects_names_from_file(NAMES);
	std::string filename = VIDEO;
	setRandomColors(Colors, totalColor);

	/* For manual input for video */
	// std::string filename;
	// std::cout << "input image or video filename: ";
	// std::cin >> filename;
	// if (filename.size() == 0) break;

	try {
#ifdef OPENCV
		std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
		if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov") {	// video file
			cv::Mat frame;
			detector.nms = 0.02;	// comment it - if track_id is not required

			cv::VideoCapture cap(filename);
			if (!cap.isOpened()) {
				std::cout << "Cannot open the video" << std::endl;
			}

			cv::Size imgsize = cv::Size(1920, 1080);

			// camera parameters
			double fx = 1473.09;
			double fy = 1487.80;
			double cx = 961.20;
			double cy = 545.38;

			cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
			cameraMatrix.at<double>(0, 0) = fx;
			cameraMatrix.at<double>(1, 1) = fy;
			cameraMatrix.at<double>(2, 2) = 1;
			cameraMatrix.at<double>(0, 2) = cx;
			cameraMatrix.at<double>(1, 2) = cy;
			cameraMatrix.at<double>(0, 1) = 0;
			cameraMatrix.at<double>(1, 0) = 0;
			cameraMatrix.at<double>(2, 0) = 0;
			cameraMatrix.at<double>(2, 1) = 0;

			std::cout << "\nInitial cameraMatrix: " << cameraMatrix << std::endl;

			// distortion coefficients
			double k1 = 0.20;
			double k2 = -0.36;

			cv::Mat distCoeffs(4, 1, cv::DataType<double>::type);
			distCoeffs.at<double>(0) = k1;
			distCoeffs.at<double>(1) = k2;
			distCoeffs.at<double>(2) = 0;
			distCoeffs.at<double>(3) = 0;

			cv::Mat distCoeffss(4, 1, cv::DataType<double>::type);
			distCoeffss.at<double>(0) = 0;
			distCoeffss.at<double>(1) = 0;
			distCoeffss.at<double>(2) = 0;
			distCoeffss.at<double>(3) = 0;

			// rotation and translation vectors
			cv::Mat rvec(3, 1, cv::DataType<double>::type);
			rvec.at<double>(0) = 1.556623851712851;
			rvec.at<double>(1) = 0.03622911742374299;
			rvec.at<double>(2) = -0.01955021028707733;

			cv::Mat tvec(3, 1, cv::DataType<double>::type);
			tvec.at<double>(0) = 0.5503291257274739;
			tvec.at<double>(1) = -0.5748104730109016;
			tvec.at<double>(2) = -0.8414083064738452;

			std::cout << "\nrvec: " << rvec << std::endl;
			std::cout << "\ntvec: " << tvec << std::endl;


			char file_name[100];

			// For the next every frame
			for (int count = 0; cap >> frame, cap.isOpened(), count < FILENUM; ++frameCnt, ++count) {
				if (frame.empty()) {
					std::cout << " ** NO VIDEO ** " << std::endl;
					break;
				}

				// The first frame
				if (frameCnt == 0) {
					WIDTH = frame.size().width;
					HEIGHT = frame.size().height;
				}

				//make undistorted map for each image
				cv::Mat map1, map2;
				initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, imgsize, CV_16SC2, map1, map2);

				//show undistorted image
				cv::Mat undistorted;
				remap(frame, frame, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());


				/* PCL */

				// PCD 파일일 때 - LOAD PCD 이용
				sprintf(file_name, "pcd\\pcd%d.pcd", count);
				std::vector<cv::Point3f> objectPoints;
				std::vector<double> distance_vec;
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::io::loadPCDFile(file_name, *cloud);

				for (int i = 0; i < cloud->size(); i++) {
					objectPoints.push_back(cv::Point3f(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z));
					double distance = sqrt(pow(objectPoints[i].x, 2) + pow(objectPoints[i].y, 2) + pow(objectPoints[i].z, 2));
					distance_vec.push_back(distance);
				}

				// TEXT 파일일 때 - 단순 파일 입출력
				/*
				sprintf(file_name, "data\\data%d.txt", count);
				std::ifstream ifs;
				ifs.open(file_name);

				float x = 0, y = 0, z = 0;
				while (!ifs.eof()) {
					ifs >> x;
					ifs >> y;
					ifs >> z;

					if (y > 0)
						objectPoints.push_back(cv::Point3f(x, y, z));
				}
				ifs.close();
				*/


				// project all lidar points to 2D plane
				std::vector<cv::Point2f> projectedPoints;
				cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffss, projectedPoints);				// erase disCoeffs


				/* YOLO */

				std::vector<bbox_t> result_vec = detector.detect(frame, minProb);
				result_vec = detector.tracking_id(result_vec, true, 8, maxDistance);	// comment it - if track_id is not required

				draw_boxes(frame, result_vec, obj_names, projectedPoints, distance_vec, 3);
				show_result(result_vec, obj_names);


				// show projected lidar points on 2D plane			
				// int r = 0;

				//while (r < projectedPoints.size()) {
				//	circle(frame, projectedPoints[r], 3, CV_RGB(255, 0, 0));											//lidarpoint - red
				//	r++;
				//}


				cv::imshow("Video", frame);
				// cv::imwrite(std::to_string(frameCnt) + ".png", frame);	// frame 단위로 저장해보기

				// projectedPoints.clear();
				// objectPoints.clear();

				if (cv::waitKey(10) >= 0) {
					std::cout << " ** VIDEO TERMINATED ** " << std::endl;
					break;
				}
			}

			cap.release();
		}
		else {	// image file
			cv::Mat mat_img = cv::imread(filename);
			std::vector<bbox_t> result_vec = detector.detect(mat_img);
			// draw_boxes(mat_img, result_vec, obj_names);
			show_result(result_vec, obj_names);
		}
#else
		//std::vector<bbox_t> result_vec = detector.detect(filename);

		auto img = detector.load_image(filename);
		std::vector<bbox_t> result_vec = detector.detect(img);
		detector.free_image(img);
		show_result(result_vec, obj_names);
#endif			
	}
	catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
	catch (...) { std::cerr << "unknown exception \n"; getchar(); }

	return 0;
}
