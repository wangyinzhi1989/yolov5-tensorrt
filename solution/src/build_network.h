#pragma once

#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "logging.h"

#define USE_FP16 // comment out this if want to use FP32
#define BATCH_SIZE (1)
#define DEVICE (0) // GPU id

#define INPUT_H (5952)
#define INPUT_W (8192)

#define NUM_ANCHORS_PER_LAYER   (3)
#define OUTPUT_C_PER_ANCHOR     (6) // for COCO: 5 + 80(num_classes)
#define OUTPUT_C    (NUM_ANCHORS_PER_LAYER * OUTPUT_C_PER_ANCHOR)
#define OUTPUT_H0   (INPUT_H / 8)
#define OUTPUT_W0   (INPUT_W / 8)
#define OUTPUT_H1   (INPUT_H / 16)
#define OUTPUT_W1   (INPUT_W / 16)
#define OUTPUT_H2   (INPUT_H / 32)
#define OUTPUT_W2   (INPUT_W / 32)

#define OUTPUT_SIZE0    (OUTPUT_C * (OUTPUT_H0 * OUTPUT_W0 + OUTPUT_H1 * OUTPUT_W1 + OUTPUT_H2 * OUTPUT_W2))
#define OUTPUT_SIZE1    (OUTPUT_C * OUTPUT_H1 * OUTPUT_W1)
#define OUTPUT_SIZE2    (OUTPUT_C * OUTPUT_H2 * OUTPUT_W2)

#define CONF_THRESH (0.4)
#define IOU_THRESH  (0.45)

#define INPUT_BLOB_NAME     ("data")
#define OUTPUT_BLOB_NAME0   ("output0")
#define OUTPUT_BLOB_NAME1   ("output1")
#define OUTPUT_BLOB_NAME2   ("output2")

extern Logger gLogger;

using namespace nvinfer1;

class CBuildNetwork
{
public:
    CBuildNetwork() = default;
    ~CBuildNetwork() = default;

    static ICudaEngine* build_network_s(unsigned int maxBatchSize, IBuilder* builder,
        IBuilderConfig* config, DataType dt);

    static ICudaEngine* build_network_m(unsigned int maxBatchSize, IBuilder* builder,
        IBuilderConfig* config, DataType dt);

    static void api_to_model(unsigned int maxBatchSize, IHostMemory** modelStream);

    static void do_inference(IExecutionContext& context, float* input,
        float* output0, int batchSize);
};

