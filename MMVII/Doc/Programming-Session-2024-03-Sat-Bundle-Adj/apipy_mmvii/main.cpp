#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace std::chrono;
using ClockType = std::chrono::steady_clock;
#define SIMPLE
int main(int argc, char** argv )
{
    cv::Mat im = cv::imread( "in.png", cv::IMREAD_GRAYSCALE );

    auto start = ClockType::now();
    uint8_t* im_data = (uint8_t*)im.data;
    
    for (int l=0; l<im.rows; ++l)
        for (int c=0; c<im.cols; ++c)
            im_data[l*im.cols+c] = im_data[l*im.cols+c] * c / im.cols;

    auto end = ClockType::now();
    auto duration = (end - start);
    auto ms = duration_cast<milliseconds>(duration).count();
    std::cout << "duration: " << ms << " ms " << '\n';

    cv::imwrite("out.png", im);
    return 0;
}
