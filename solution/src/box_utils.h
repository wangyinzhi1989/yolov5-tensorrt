#ifndef BOX_UTILS_H_
#define BOX_UTILS_H_

#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;

typedef vector<vector<vector<float> > > vector3f;
typedef vector<vector<vector<vector<vector<float> > > > > vector5f;

struct BoxInfo {
    BoxInfo(const vector<float>& _box, float _score, int _cls_id) {
        box = _box;
        score = _score;
        cls_id = _cls_id;
    }
    vector<float> box;
    float score;
    int cls_id;
};

inline float sigmoid(float x);

cv::Mat preprocess_img(const cv::Mat& img, int ih, int iw);

// only single img per batch implemented
void img2input(const cv::Mat& img, vector<float>& data);

cv::Rect restore_coords(const cv::Mat& img, const vector<float>& bbox, int ih, int iw);

void init_output_cache(int batch_size, int num_anchors,
    int height, int width, int output_c_per_anchor, vector5f& output);

vector<int> nms_kernel(const vector<vector<float> >& boxes,
    const vector<float>& scores, float iou_thresh);

void post_process(const float* data, int data_size, int batch_size, int output_c_per_anchor,
    int height, int width, int stride, float conf_thresh, const vector<vector<float> >& anchors,
    vector5f& output_cache, vector3f& boxes);

// shape of boxes_wh: [batch_size, num_boxes, output_c_per_anchor]
void non_max_suppression(float conf_thresh, float iou_thresh,
    vector3f& boxes_wh, vector<vector<BoxInfo> >& filtered_boxes);

#endif
