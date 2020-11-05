#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <algorithm>

#define OPENCV

#include "yolo_v2_class.hpp"	// imported functions from DLL

#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#pragma comment(lib, "opencv_world341.lib") 
#else
#pragma comment(lib, "opencv_core2413.lib")  
#pragma comment(lib, "opencv_imgproc2413.lib")  
#pragma comment(lib, "opencv_highgui2413.lib") 
#endif

int WIDTH;
int HEIGHT;

const int PERSON = 0;		const int BICYCLE = 1;		const int CAR = 2;
const int MOTORBIKE = 3;	const int BUS = 5;			const int TRUCK = 7;
const int BIRD = 14;		const int CAT = 15;			const int DOG = 16;
const std::vector<int> Objects = { PERSON, BICYCLE, CAR, MOTORBIKE, BUS, TRUCK, BIRD, CAT, DOG };


		/* Pre-setting for detecting */
//////////////////////////////////////////////////
												//
const std::string CFG = "yolov3.cfg";			//
const std::string WEIGHTS = "yolov3.weights";	//
const std::string NAMES = "coco.names";			//
const std::string VIDEO = "video.avi";			//
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
const int maxDistance = 150;					//
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
		// When Distance is smaller than maxDistance, and frame difference is smaller than 10
		if (euclideanDist(prevPos, currPos) < maxDistance && prevFrame + 30 > currFrame) {
			// std::cout << " ** " << euclideanDist(prevPos, currPos) << " ** " << std::endl;
			// std::cout << " ** " << prevFrame << " " << currFrame << " ** " << std::endl;

			bool left = currPos.x < WIDTH / 2 && currPos.x - prevPos.x > 5;
			bool right = currPos.x > WIDTH / 2 && prevPos.x - currPos.x > 5;
			bool bigger = obj.w - prevObj.w >= 2 && obj.h - prevObj.h >= 2;
			
			// std::cout << (left ? "LEFT " : "") << (right ? "RIGHT " : "") << (bigger ? "BIGGER " : "") << std::endl;

			if (left || right || bigger) {
				int bbox_w = (obj.x + obj.w < WIDTH) ? obj.w : WIDTH - obj.x;
				int bbox_h = (obj.y + obj.h < HEIGHT) ? obj.h : HEIGHT - obj.y;
				// std::cout << bbox_w << " & " << bbox_h << std::endl;
				
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


void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, unsigned int wait_msec = 0) {
	for (auto &i : result_vec) {

		bool condition = std::find(Objects.begin(), Objects.end(), i.obj_id) != Objects.end() && i.prob > minProb;
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
	cv::imshow("window name", mat_img);
	// cv::waitKey(wait_msec);
}
#endif	// OPENCV


void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
	for (auto &i : result_vec) {

		bool condition = std::find(Objects.begin(), Objects.end(), i.obj_id) != Objects.end() && i.prob > minProb;
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
			
			// For the next every frame
			for (cv::VideoCapture cap(filename); cap >> frame, cap.isOpened(); ++frameCnt) {
				if (frame.empty()) {
					std::cout << " ** NO VIDEO ** " << std::endl;
					break;
				}

				// The first frame
				if (frameCnt == 0) {
					WIDTH = frame.size().width;
					HEIGHT = frame.size().height;

					std::cout << " ** Size of " << WIDTH << " * " << HEIGHT << " video ** " << std::endl;
				}

				std::vector<bbox_t> result_vec = detector.detect(frame, 0.2);
				result_vec = detector.tracking_id(result_vec, true, 8, maxDistance);	// comment it - if track_id is not required

				draw_boxes(frame, result_vec, obj_names, 3);
				show_result(result_vec, obj_names);

				if (cv::waitKey(10) >= 0) {
					std::cout << " ** VIDEO TERMINATED ** " << std::endl;
					break;
				}
			}
		}
		else {	// image file
			cv::Mat mat_img = cv::imread(filename);
			std::vector<bbox_t> result_vec = detector.detect(mat_img);
			draw_boxes(mat_img, result_vec, obj_names);
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
