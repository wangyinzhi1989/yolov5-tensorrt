#include <chrono>
#include <opencv2/opencv.hpp>
#include "box_utils.h"

float sigmoid(float x) {
    return (1.f / (1.f + exp(-x)));
}
//
//cv::Mat preprocess_img(const cv::Mat& img, int ih, int iw) {
//    int w, h, x, y;
//    float r_w = 1.f * iw / img.cols;
//    float r_h = 1.f * ih / img.rows;
//    if (r_h > r_w) {
//        w = iw;
//        h = int(r_w * img.rows);
//        x = 0;
//        y = int((ih - h) / 2);
//    } else {
//        w = int(r_h * img.cols);
//        h = ih;
//        x = int((iw - w) / 2);
//        y = 0;
//    }
//    cv::Mat re(h, w, CV_8UC3);
//    cv::resize(img, re, re.size());
//    cv::cvtColor(re, re, cv::COLOR_BGR2RGB);
//    cv::Mat out(ih, iw, CV_8UC3, cv::Scalar(114, 114, 114));
//    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
//    out.convertTo(out, CV_32F);
//    out /= 255.f;
//    return out;
//}

cv::Mat preprocess_img(const cv::Mat& img, int ih, int iw) {
    int w, h, x, y;
    float r_w = 1.f * iw / img.cols;
    float r_h = 1.f * ih / img.rows;
    auto r = std::min(r_w, r_h);
    w = r * img.cols;
    h = r * img.rows;
    int dw = (iw - w) % 64, dh = (ih - h) % 64;
    x = round(dw / 2.0);
    y = round(dh / 2.0);

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::cvtColor(re, re, cv::COLOR_BGR2RGB);
    cv::Mat out(h + dh, w + dw, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    out.convertTo(out, CV_32F);
    out /= 255.f;
    return out;
}

// only single img per batch implemented
void img2input(const cv::Mat& img, vector<float>& data) {
    // data must have been resized before
    assert(data.size() == (img.rows * img.cols * img.channels()));

    int HEIGHT = img.rows;
    int WIDTH = img.cols;
    for (size_t i = 0; i < HEIGHT; i++) {
        for (size_t j = 0; j < WIDTH; j++) {
            cv::Vec3f pixel = img.at<cv::Vec3f>(i, j);
            data[0 * HEIGHT * WIDTH + i * WIDTH + j] = pixel[0];
            data[1 * HEIGHT * WIDTH + i * WIDTH + j] = pixel[1];
            data[2 * HEIGHT * WIDTH + i * WIDTH + j] = pixel[2];
        }
    }
}

cv::Rect restore_coords(const cv::Mat& img, const vector<float>& bbox, int ih, int iw) {
    assert(bbox.size() == 4);

    float x1 = bbox[0];
    float y1 = bbox[1];
    float x2 = bbox[2];
    float y2 = bbox[3];

    float l, r, t, b;
    float r_w = 1.f * iw / img.cols;
    float r_h = 1.f * ih / img.rows;
    //cout << "r_w: " << r_w << endl;
    //cout << "r_h: " << r_h << endl;
    if (r_h > r_w) {
        l = x1 / r_w;
        r = x2 / r_w;
        t = (y1 - (ih - r_w * img.rows) / 2.f) / r_w;
        b = (y2 - (ih - r_w * img.rows) / 2.f) / r_w;
    } else {
        l = (x1 - (iw - r_h * img.cols) / 2.f) / r_h;
        r = (x2 - (iw - r_h * img.cols) / 2.f) / r_h;
        t = y1 / r_h;
        b = y2 / r_h;
    }
    return cv::Rect(int(l), int(t), int(r-l), int(b-t));
}

void init_output_cache(int batch_size, int num_anchors,
    int height, int width, int output_c_per_anchor, vector5f& output) {
    output.resize(batch_size);
    for (size_t b = 0; b < batch_size; b++) {
        output[b].resize(num_anchors);
    }
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t a = 0; a < num_anchors; a++) {
            output[b][a].resize(height);
        }
    }
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t a = 0; a < num_anchors; a++) {
            for (size_t h = 0; h < height; h++) {
                output[b][a][h].resize(width);
            }
        }
    }
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t a = 0; a < num_anchors; a++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    output[b][a][h][w].resize(output_c_per_anchor);
                }
            }
        }
    }
}

vector<int> nms_kernel(const vector<vector<float> >& boxes,
    const vector<float>& scores, float iou_thresh) {
    typedef std::pair<BoxInfo, int> BoxIndex;
    vector<BoxIndex> sorted_boxes;
    vector<int> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); i++) {
        sorted_boxes.emplace_back(std::make_pair(BoxInfo(boxes[i], scores[i], -1), i)); // cls_id ignored
        indices[i] = i;
    }
    std::sort(sorted_boxes.begin(), sorted_boxes.end(),
        [](const BoxIndex& x, const BoxIndex& y) {return x.first.score > y.first.score; });
    /*
    // print
    cout << "################################################" << endl;
    for (size_t i = 0; i < sorted_boxes.size(); i++) {
        cout << sorted_boxes[i].second << " "
             << sorted_boxes[i].first.score << " ("
             << sorted_boxes[i].first.box[0] << ", "
             << sorted_boxes[i].first.box[1] << ", "
             << sorted_boxes[i].first.box[2] << ", "
             << sorted_boxes[i].first.box[3] << ")" << endl;
    }
    cout << "################################################" << endl;
    */

    vector<int> kept;
    while (indices.size() > 0) {
        int good_idx = indices[0];
        //cout << "good_idx: " << good_idx << endl;
        kept.push_back(sorted_boxes[good_idx].second);
        vector<int> tmp = indices;
        indices.clear();
        for (size_t i = 1; i < tmp.size(); i++) {
            int tmp_i = tmp[i];
            vector<float> good_box = sorted_boxes[good_idx].first.box;
            vector<float> tmp_box = sorted_boxes[tmp_i].first.box;
            float inter_x1 = std::max(good_box[0], tmp_box[0]);
            float inter_y1 = std::max(good_box[1], tmp_box[1]);
            float inter_x2 = std::min(good_box[2], tmp_box[2]);
            float inter_y2 = std::min(good_box[3], tmp_box[3]);
            float inter_w = std::max((inter_x2 - inter_x1), 0.f);
            float inter_h = std::max((inter_y2 - inter_y1), 0.f);
            float inter_area = inter_w * inter_h;
            float area_1 = (good_box[2] - good_box[0]) * (good_box[3] - good_box[1]);
            float area_2 = (tmp_box[2] - tmp_box[0]) * (tmp_box[3] - tmp_box[1]);
            float iou = inter_area / (area_1 + area_2 - inter_area);
            if (iou <= iou_thresh) {
                indices.push_back(tmp_i);
            }
        }
    }

    return kept;
}
//
//void post_process(const float* data, int data_size, int batch_size, int output_c_per_anchor,
//    int height, int width, int stride, float conf_thresh, const vector<vector<float> >& anchors,
//    vector5f& output_cache, vector3f& boxes) {
//    // examaples from COCO
//    const int num_anchors = int(anchors.size());      // 3
//    int output_c = num_anchors * output_c_per_anchor; // 3 * 85
//    assert(data_size == (batch_size * output_c * height * width));
//
//    // flattened array [batch_size, output_c, height, width]         ==(view)==>
//    // [batch_size, num_anchors, output_c_per_anchor, height, width] ==(permute)==>
//    // [batch_size, num_anchors, height, width, output_c_per_anchor]
//
//    int step0 = num_anchors * height * width * output_c_per_anchor;
//    int step1 = height * width * output_c_per_anchor;
//    int step2 = width * output_c_per_anchor;
//    for (size_t b = 0; b < batch_size; b++) {                               // batch_size
//        for (size_t a = 0; a < num_anchors; a++) {                          // num_anchors
//            for (size_t h = 0; h < height; h++) {                           // height
//                for (size_t w = 0; w < width; w++) {                        // width
//                    int offset0 = b * step0 + a * step1 + h * step2 + w * output_c_per_anchor + 4;
//                    float y0 = sigmoid(data[offset0]);
//                    if (y0 < conf_thresh) {
//                        output_cache[b][a][h][w][4] = 0.f;
//                        continue;
//                    }
//                    for (size_t o = 0; o < output_c_per_anchor; o++) {      // output_c_per_anchor
//                        int offset = b * step0 + a * step1 + h * step2 + w * output_c_per_anchor + o;
//                        float y = sigmoid(data[offset]);
//                        if (o >= 4) {
//                            output_cache[b][a][h][w][o] = y;                                 // [4:]
//                        } else if (o == 0) {
//                            output_cache[b][a][h][w][o] = (y * 2.f - 0.5f + w) * stride;     // center_x
//                        } else if (o == 1) {
//                            output_cache[b][a][h][w][o] = (y * 2.f - 0.5f + h) * stride;     // center_y
//                        } else if (o == 2) {
//                            output_cache[b][a][h][w][o] = pow(y * 2.f, 2.f) * anchors[a][0]; // w
//                        } else if (o == 3) {
//                            output_cache[b][a][h][w][o] = pow(y * 2.f, 2.f) * anchors[a][1]; // h
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    // reshape to [batch_size, -1, output_c_per_anchor]
//    for (size_t b = 0; b < batch_size; b++) {
//        for (size_t a = 0; a < num_anchors; a++) {
//            for (size_t h = 0; h < height; h++) {
//                for (size_t w = 0; w < width; w++) {
//                    if (output_cache[b][a][h][w][4] >= conf_thresh) {
//                        // do not use std::move() here
//                        boxes[b].emplace_back(output_cache[b][a][h][w]);
//                    }
//                }
//            }
//        }
//    }
//}

void post_process(const float* data, int data_size, int batch_size, int output_c_per_anchor,
    int height, int width, int stride, float conf_thresh, const vector<vector<float> >& anchors,
    vector5f& output_cache, vector3f& boxes) {
    // examaples from COCO
    const int num_anchors = int(anchors.size());      // 3
    int output_c = num_anchors * output_c_per_anchor; // 3 * 85
    assert(data_size == (batch_size * output_c * height * width));

    // flattened array [batch_size, output_c, height, width]         ==(view)==>
    // [batch_size, num_anchors, output_c_per_anchor, height, width] ==(permute)==>
    // [batch_size, num_anchors, height, width, output_c_per_anchor]

    int step0 = num_anchors * height * width * output_c_per_anchor;
    int step1 = height * width * output_c_per_anchor;
    int step2 = width * output_c_per_anchor;
    for (size_t b = 0; b < batch_size; b++) {                               // batch_size
        for (size_t a = 0; a < num_anchors; a++) {                          // num_anchors
            for (size_t h = 0; h < height; h++) {                           // height
                for (size_t w = 0; w < width; w++) {                        // width
                    int offset0 = b * step0 + a * step1 + h * step2 + w * output_c_per_anchor + 4;
                    float y0 = sigmoid(data[offset0]);
                    if (y0 < conf_thresh) {
                        output_cache[b][a][h][w][4] = 0.f;
                        continue;
                    }
                    for (size_t o = 0; o < output_c_per_anchor; o++) {      // output_c_per_anchor
                        int offset = b * step0 + a * step1 + h * step2 + w * output_c_per_anchor + o;
                        float y = sigmoid(data[offset]);
                        if (o >= 4) {
                            output_cache[b][a][h][w][o] = y;                                 // [4:]
                        }
                        else if (o == 0) {
                            output_cache[b][a][h][w][o] = (y * 2.f - 0.5f + w) * stride;     // center_x
                        }
                        else if (o == 1) {
                            output_cache[b][a][h][w][o] = (y * 2.f - 0.5f + h) * stride;     // center_y
                        }
                        else if (o == 2) {
                            output_cache[b][a][h][w][o] = pow(y * 2.f, 2.f) * anchors[a][0]; // w
                        }
                        else if (o == 3) {
                            output_cache[b][a][h][w][o] = pow(y * 2.f, 2.f) * anchors[a][1]; // h
                        }
                    }
                }
            }
        }
    }

    // reshape to [batch_size, -1, output_c_per_anchor]
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t a = 0; a < num_anchors; a++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    if (output_cache[b][a][h][w][4] >= conf_thresh) {
                        // do not use std::move() here
                        boxes[b].emplace_back(output_cache[b][a][h][w]);
                    }
                }
            }
        }
    }
}

// shape of boxes_wh: [batch_size, num_boxes, output_c_per_anchor]
void non_max_suppression(float conf_thresh, float iou_thresh,
    vector3f& boxes_wh, vector<vector<BoxInfo> >& filtered_boxes) {
    filtered_boxes.clear();
    filtered_boxes.resize(boxes_wh.size()); // batch_size
    int max_wh = 5000; // stride, makes boxes of each class_id nonoverlapping when nms
    for (size_t b = 0; b < boxes_wh.size(); b++) {
        vector<vector<float> > boxes; // x1, y1, x2, y2
        vector<vector<float> > boxes_stride;
        vector<float> scores;
        vector<int> cls_ids;
        for (size_t i = 0; i < boxes_wh[b].size(); i++) {
            boxes_wh[b][i][0] -= (boxes_wh[b][i][2] / 2.f); // x1 = center_x - (w / 2)
            boxes_wh[b][i][1] -= (boxes_wh[b][i][3] / 2.f); // y1 = center_y - (h / 2)
            boxes_wh[b][i][2] += boxes_wh[b][i][0];         // x2 = w + x1
            boxes_wh[b][i][3] += boxes_wh[b][i][1];         // y2 = h + y1
            for (size_t j = 5; j < boxes_wh[b][i].size(); j++) {
                boxes_wh[b][i][j] *= boxes_wh[b][i][4];
                // filtered by thresh
                if (boxes_wh[b][i][j] >= conf_thresh) {
                    vector<float> t(boxes_wh[b][i].begin(), boxes_wh[b][i].begin() + 4);
                    boxes.emplace_back(t);
                    for (size_t k = 0; k < t.size(); k++) {
                        t[k] += (j * max_wh);
                    }
                    boxes_stride.emplace_back(t);
                    scores.push_back(boxes_wh[b][i][j]);
                    cls_ids.push_back(j); // start from 5
                }
            }
        }
        //cout << "boxes.size(): " << boxes.size() << endl;

        vector<int> kept = nms_kernel(boxes_stride, scores, iou_thresh);
        //cout << "kept.size(): " << kept.size() << endl;
        for (size_t k = 0; k < kept.size(); k++) {
            //cout << "kept[" << k << "]: " << kept[k] << endl;
            filtered_boxes[b].emplace_back(BoxInfo(boxes[kept[k]], scores[kept[k]], cls_ids[kept[k]] - 5));
        }
    }
}
