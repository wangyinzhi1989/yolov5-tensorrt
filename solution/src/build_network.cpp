#include "yololayer.h"
#include "common.h"
#include "box_utils.h"
#include "build_network.h"

Logger gLogger;

ICudaEngine* CBuildNetwork::build_network_s(unsigned int maxBatchSize, IBuilder* builder,
    IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<string, Weights> weightMap = loadWeights("E:\\ObjectDetection\\yolov5-tensorrtapi-test\\solution\\res\\qrcode-s.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, INPUT_H, INPUT_W, "model.0");
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 1, false, 1, 0.5, "model.9");

    // yolov5 head
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{ 2, 2 },
        weightMap["model.11.ConvTranspose2d.weight"], emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(256);

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{ 2, 2 },
        weightMap["model.15.ConvTranspose2d.weight"], emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(128);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 128, 1, false, 1, 0.5, "model.17");
    // output0
    IConvolutionLayer* conv24_0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), OUTPUT_C, DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv18 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 256, 1, false, 1, 0.5, "model.20");
    // output1
    IConvolutionLayer* conv24_1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), OUTPUT_C, DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBnLeaky(network, weightMap, *bottleneck_csp20->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor* inputTensors24[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors24, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 512, 1, false, 1, 0.5, "model.23");
    // output2
    IConvolutionLayer* conv24_2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), OUTPUT_C, DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    /*conv24_0->getOutput(0)->setName(OUTPUT_BLOB_NAME0);
    conv24_1->getOutput(0)->setName(OUTPUT_BLOB_NAME1);
    conv24_2->getOutput(0)->setName(OUTPUT_BLOB_NAME2);
    network->markOutput(*conv24_0->getOutput(0));
    network->markOutput(*conv24_1->getOutput(0));
    network->markOutput(*conv24_2->getOutput(0));*/

    auto yolo = addYoLoLayer(network, weightMap, conv24_0, conv24_1, conv24_2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME0);
    network->markOutput(*yolo->getOutput(0));

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(800*(1 << 20)); // 16MB
#ifdef USE_FP16
    std::cout<<"use fp16"<<std::endl;
    config->setFlag(BuilderFlag::kFP16);
#endif
    cout << "building engine, please wait for a while..." << endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (nullptr != engine)
        cout << "build engine successfully!" << endl;
    else
    {
        cout << "build engine failed!" << endl;
        abort();
    }

    network->destroy();

    // release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* CBuildNetwork::build_network_m(unsigned int maxBatchSize, IBuilder* builder,
    IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<string, Weights> weightMap = loadWeights("/home/wangyinzhi/work/from_wanglin/yolov5-tensorrtApi/model/yolov5m.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 64, 3, INPUT_H, INPUT_W, "model.0");
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), 128, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 128, 3, true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), 256, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 256, 9, true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), 512, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 512, 9, true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), 1024, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1024, 1024, 5, 9, 13, "model.8");
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1024, 3, false, 1, 0.5, "model.9");

    // yolov5 head
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), 512, 1, 1, 1, "model.10");

    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 512, DimsHW{ 2, 2 },
        weightMap["model.11.ConvTranspose2d.weight"], emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(512);

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 3, false, 1, 0.5, "model.13");
    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), 256, 1, 1, 1, "model.14");

    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 256, DimsHW{ 2, 2 },
        weightMap["model.15.ConvTranspose2d.weight"], emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(256);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 3, false, 1, 0.5, "model.17");
    // output0
    IConvolutionLayer* conv24_0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), OUTPUT_C, DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    IShuffleLayer* sf0 = network->addShuffle(*conv24_0->getOutput(0));
    Dims dims;
    dims.nbDims = 5;
    dims.d[0] = BATCH_SIZE;
    dims.d[1] = NUM_ANCHORS_PER_LAYER;
    dims.d[2] = OUTPUT_C_PER_ANCHOR;
    dims.d[3] = OUTPUT_H0;
    dims.d[4] = OUTPUT_W0;
    sf0->setReshapeDimensions(dims);
    // setFirstTranspose(): before reshape
    // setSecondTranspose(): after reshape
    sf0->setSecondTranspose(Permutation{ 0, 1, 3, 4, 2 });

    auto conv18 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), 256, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 512, 3, false, 1, 0.5, "model.20");
    // output1
    IConvolutionLayer* conv24_1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), OUTPUT_C, DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    IShuffleLayer* sf1 = network->addShuffle(*conv24_1->getOutput(0));
    dims.d[3] = OUTPUT_H1;
    dims.d[4] = OUTPUT_W1;
    sf1->setReshapeDimensions(dims);
    sf1->setSecondTranspose(Permutation{ 0, 1, 3, 4, 2 });

    auto conv21 = convBnLeaky(network, weightMap, *bottleneck_csp20->getOutput(0), 512, 3, 2, 1, "model.21");
    ITensor* inputTensors24[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors24, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 1024, 3, false, 1, 0.5, "model.23");
    // output2
    IConvolutionLayer* conv24_2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), OUTPUT_C, DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);
    IShuffleLayer* sf2 = network->addShuffle(*conv24_2->getOutput(0));
    dims.d[3] = OUTPUT_H2;
    dims.d[4] = OUTPUT_W2;
    sf2->setReshapeDimensions(dims);
    sf2->setSecondTranspose(Permutation{ 0, 1, 3, 4, 2 });

    sf0->getOutput(0)->setName(OUTPUT_BLOB_NAME0);
    sf1->getOutput(0)->setName(OUTPUT_BLOB_NAME1);
    sf2->getOutput(0)->setName(OUTPUT_BLOB_NAME2);
    network->markOutput(*sf0->getOutput(0));
    network->markOutput(*sf1->getOutput(0));
    network->markOutput(*sf2->getOutput(0));

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize((1 << 30)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    cout << "building engine, please wait for a while..." << endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    cout << "build engine successfully!" << endl;

    network->destroy();

    // release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void CBuildNetwork::api_to_model(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = build_network_s(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = build_network_m(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine
    (*modelStream) = engine->serialize();

    // close everything down
    engine->destroy();
    builder->destroy();
}

void CBuildNetwork::do_inference(IExecutionContext& context, float* input,
    float* output0, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // pointers to input and output device buffers to pass to engine
    // engine requires exactly IEngine::getNbBindings() number of buffers
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // in order to bind the buffers, we need to know the names of the input and output tensors
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0);
    //const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);
    //const int outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
    //cout << "inputIndex: " << inputIndex << endl;
    //cout << "outputIndex0: " << outputIndex0 << endl;
    //cout << "outputIndex1: " << outputIndex1 << endl;
    //cout << "outputIndex2: " << outputIndex2 << endl;

    // create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * OUTPUT_SIZE0 * sizeof(float)));
 /*   CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * OUTPUT_SIZE2 * sizeof(float)));*/

    // create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
        batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output0, buffers[outputIndex0],
        batchSize * OUTPUT_SIZE0 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    /*CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1],
        batchSize * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output2, buffers[outputIndex2],
        batchSize * OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost, stream));*/
    cudaStreamSynchronize(stream);

    // release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    //CHECK(cudaFree(buffers[outputIndex1]));
    //CHECK(cudaFree(buffers[outputIndex2]));
}
