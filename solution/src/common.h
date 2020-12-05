#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <map>
#include <cmath>
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


using namespace nvinfer1;
using namespace std;


// tensorRT weights file have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<string, Weights> loadWeights(const string& file);

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<string, Weights>& weightMap,
    ITensor& input, const string& lname, float eps);

ILayer* convBnLeaky(INetworkDefinition* network, std::map<string, Weights>& weightMap,
    ITensor& input, int outch, int ksize, int s, int g, const string& lname);

ILayer* focus(INetworkDefinition* network, std::map<string, Weights>& weightMap,
    ITensor& input, int inch, int outch, int ksize, int ih, int iw, const string& lname);

ILayer* bottleneck(INetworkDefinition* network, std::map<string, Weights>& weightMap,
    ITensor& input, int c1, int c2, bool shortcut, int g, float e, const string& lname);

ILayer* bottleneckCSP(INetworkDefinition* network, std::map<string, Weights>& weightMap,
    ITensor& input, int c2, int n, bool shortcut, int g, float e, const string& lname);

ILayer* SPP(INetworkDefinition* network, std::map<string, Weights>& weightMap,
    ITensor& input, int c1, int c2, int k1, int k2, int k3, const string& lname);

std::vector<float> getAnchors(std::map<std::string, Weights>& weightMap);

IPluginV2Layer* addYoLoLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, IConvolutionLayer* det0, IConvolutionLayer* det1, IConvolutionLayer* det2);

float iou(float lbox[4], float rbox[4]);

inline bool cmp(const Yolo::Detection& a, const Yolo::Detection& b);

void nms(std::vector<Yolo::Detection>& res, float* output, float conf_thresh, float nms_thresh = 0.5);
cv::Rect get_rect(cv::Mat& img, float bbox[4]);
#endif
