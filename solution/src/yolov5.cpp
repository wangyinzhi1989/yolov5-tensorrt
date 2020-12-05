#include <chrono>
#include <fstream>
#include "build_network.h"
#include "box_utils.h"
#include "common.h"

using namespace nvinfer1;
using namespace std;


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{nullptr};
    size_t size{0};
    string engine_name = "E:\\ObjectDetection\\yolov5-tensorrtapi-test\\solution\\res\\demo-8192-5952.engine";
    string img_path = "";
    if (argc == 2 && string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        CBuildNetwork::api_to_model(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && string(argv[1]) == "-d") {
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        img_path = argv[2];
    } else {
        std::cerr << "arguments not right!" << endl;
        std::cerr << "./yolov5 -s  // serialize model to plan file" << endl;
        std::cerr << "./yolov5 -d test.jpg  // deserialize plan file and run inference" << endl;
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;


    // ######## prepare input data and output cache ########
    vector<float> input_data;
    input_data.resize(BATCH_SIZE * 3 * INPUT_H * INPUT_W);
    static float output0[BATCH_SIZE * (OUTPUT_SIZE0)];
   /* static float output1[BATCH_SIZE * OUTPUT_SIZE1];
    static float output2[BATCH_SIZE * OUTPUT_SIZE2];*/
    vector5f output_cache0, output_cache1, output_cache2;
    init_output_cache(BATCH_SIZE, NUM_ANCHORS_PER_LAYER, OUTPUT_H0, OUTPUT_W0, OUTPUT_C_PER_ANCHOR, output_cache0);
    init_output_cache(BATCH_SIZE, NUM_ANCHORS_PER_LAYER, OUTPUT_H1, OUTPUT_W1, OUTPUT_C_PER_ANCHOR, output_cache1);
    init_output_cache(BATCH_SIZE, NUM_ANCHORS_PER_LAYER, OUTPUT_H2, OUTPUT_W2, OUTPUT_C_PER_ANCHOR, output_cache2);
    /*
    // data from txt file are definitely same for both pytorch and tensorRT
    std::ifstream ifs("input.txt");
    int index = 0;
    while (!ifs.eof()) {
        ifs >> input_data[index];
        index++;
    }
    cout << "input data counts: " << index << endl;
    ifs.close();
    */
    std::vector<std::vector<Yolo::Detection>> batch_res(1);
    cv::Mat img = cv::imread(img_path);
    cv::Mat tmp_pre_input = preprocess_img(img, INPUT_H, INPUT_W);
    img2input(tmp_pre_input, input_data);
    CBuildNetwork::do_inference(*context, input_data.data(), output0, BATCH_SIZE);

    auto start = std::chrono::system_clock::now();
    for (size_t i = 0; i < 100; i++) {
        batch_res.clear();
        if (!img.data) {
            cout << "img empty" << endl;
            return -1;
        }
        cv::Mat pre_input = preprocess_img(img, INPUT_H, INPUT_W);
        img2input(pre_input, input_data);

        // ######## run inference ########
        // warming up
        CBuildNetwork::do_inference(*context, input_data.data(), output0, BATCH_SIZE);

        for (int b = 0; b < 1; b++) {
            auto& res = batch_res[b];
            nms(res, &output0[b * OUTPUT_SIZE0], CONF_THRESH, IOU_THRESH);
        }
    }
    auto end = std::chrono::system_clock::now();
    cout << "times inference: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;

    for (int b = 0; b < 1; b++) {
        auto& res = batch_res[b];
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string(res[j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
    }

    cv::imwrite("E:\\data\\yolov5-qrcode\\test\\res\\paste_img_8192_5952.jpg", img);

    // destroy engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

