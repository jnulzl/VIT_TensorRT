//
// Created by lizhaoliang-os on 2020/6/9.
//
#include <cstring>
#include <npp.h>
#include "tensorrt/Module_cls_tensorrt_impl.h"
#include "logger.h"
#include "common.h"

#include "alg_define.h"
#include "debug.h"

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
#if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs = std::filesystem;
#endif
#endif
#ifndef GHC_USE_STD_FS

#include <ghc/filesystem.hpp>

namespace fs = ghc::filesystem;
#endif

namespace tensorrt_cls
{

    CModule_cls_tensorrt_impl::CModule_cls_tensorrt_impl()
    {

    }

    CModule_cls_tensorrt_impl::~CModule_cls_tensorrt_impl()
    {

    }

    void CModule_cls_tensorrt_impl::engine_deinit()
    {
        cudaFastFree(dst_ptr_d_, config_.device_id);
        cudaFastFree(dst_float_ptr_d_, config_.device_id);
        cudaFastFree(dst_chw_float_ptr_d_, config_.device_id);
        for (int bs = 0; bs < config_.batch_size; ++bs)
        {
            cudaFastFree(src_ptrs_d_[bs], config_.device_id);
        }
        cudaFastFree(coeffs_d_, config_.device_id);
        cudaFastFree(pBatchList_d_, config_.device_id);
#ifdef AI_ALG_DEBUG
        AIALG_PRINT("%d, CModule_cls_tensorrt_impl::engine_deinit\n", __LINE__);
#endif
    }

    void CModule_cls_tensorrt_impl::engine_init()
    {
        CUDACHECK(cudaSetDevice(config_.device_id));

        // config_.weights_path is onnx model
        std::string engine_file_path = config_.weights_path.substr(0, config_.weights_path.size() - 4) + "trt";
        if ("trt" == config_.weights_path.substr(config_.weights_path.size() - 3, 3))
        {
            engine_file_path = config_.weights_path;
        }
        if (!fs::exists(engine_file_path))
        {
            engine_file_path = config_.weights_path + ".trt";
        }
        if (fs::exists(engine_file_path))
        {
#ifdef AI_ALG_DEBUG
            std::cout << "Using TensorRT engine : " << engine_file_path << std::endl;
#endif
            std::ifstream engineFile(engine_file_path, std::ios::binary);
            if (!engineFile)
            {
                std::cerr << "Error opening engine file: " << engine_file_path << std::endl;
            }

            engineFile.seekg(0, engineFile.end);
            long int fsize = engineFile.tellg();
            engineFile.seekg(0, engineFile.beg);

            std::vector<char> engineData(fsize);
            engineFile.read(engineData.data(), fsize);
            if (!engineFile)
            {
                std::cerr << "Error loading engine file: " << engine_file_path << std::endl;
            }
            
            runtime_ = TRTUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));             
            if (config_.dlaCore != -1)
            {
                runtime_->setDLACore(config_.dlaCore);
            }

            net_ = std::shared_ptr<nvinfer1::ICudaEngine>(
                    runtime_->deserializeCudaEngine(engineData.data(), engineData.size()),
                    samplesCommon::InferDeleter());
        } else
        {
            builder_ = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
            if (!builder_)
            {
                AIALG_ASSERT(0);
            }

            const auto explicitBatch =
                    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            network_ = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder_->createNetworkV2(explicitBatch));
            if (!network_)
            {
                AIALG_ASSERT(0);
            }

            config_trt_ = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
            if (!config_trt_)
            {
                AIALG_ASSERT(0);
            }

            parser_ = TRTUniquePtr<nvonnxparser::IParser>(
                    nvonnxparser::createParser(*network_, sample::gLogger.getTRTLogger()));

            if (!parser_)
            {
                AIALG_ASSERT(0);
            }

            auto constructed = constructNetwork(builder_, network_, config_trt_, parser_);
            if (!constructed)
            {
                AIALG_ASSERT(0);
            }

#if NV_TENSORRT_MAJOR >= 8
            // CUDA stream used for profiling by the builder.
            auto profileStream = samplesCommon::makeCudaStream();
            if (!profileStream)
            {
                AIALG_ASSERT(0);
            }
            config_trt_->setProfileStream(*profileStream);

            TRTUniquePtr<nvinfer1::IHostMemory> plan{builder_->buildSerializedNetwork(*network_, *config_trt_)};
            if (!plan)
            {
                AIALG_ASSERT(0);
            }

            runtime_ = TRTUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
            if (!runtime_)
            {
                AIALG_ASSERT(0);
            }

            net_ = std::shared_ptr<nvinfer1::ICudaEngine>(
                    runtime_->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
            if (!net_)
            {
                AIALG_ASSERT(0);
            }
#else
            net_ = std::shared_ptr<nvinfer1::ICudaEngine>(
                    builder_->buildEngineWithConfig(*network_, *config_trt_), samplesCommon::InferDeleter());

            if (!net_)
            {
                AIALG_ASSERT(0);
            }
#endif
            // serialize() to disk for fast loader later
            if (!fs::exists(engine_file_path))
            {
                std::ofstream engineFile(engine_file_path, std::ios::binary);
                if (!engineFile)
                {
                    std::cerr << "Cannot open engine file: " << engine_file_path << std::endl;
                }

                TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine{net_->serialize()};
                if (!serializedEngine)
                {
                    std::cerr << "Engine serialization failed" << std::endl;
                }
#ifdef AI_ALG_DEBUG
                std::cout << "Writing TensorRT engine to " << engine_file_path << std::endl;
#endif
                engineFile.write(static_cast<char *>(serializedEngine->data()), serializedEngine->size());
                engineFile.close();
#ifdef AI_ALG_DEBUG
                std::cout << "Writing TensorRT engine finished!" << std::endl;
#endif
            }
        }
        assert(network_->getNbInputs() == config_.input_names.size());
//    mInputDims = network->getInput(0)->getDimensions();
//    assert(mInputDims.nbDims == 4);

        assert(network_->getNbOutputs() == config_.output_names.size());
//    mOutputDims = network->getOutput(0)->getDimensions();
//    assert(mOutputDims.nbDims == 2);

        // Create RAII buffer manager object
        buffers_ = std::make_unique<samplesCommon::BufferManager>(net_, config_.batch_size);
        context_ = TRTUniquePtr<nvinfer1::IExecutionContext>(net_->createExecutionContext());
        if (!context_)
        {
            AIALG_ASSERT(0);
        }

        // For gpu preprocess
        /************Device memory allocator and initialization***********/
        dst_pixel_num_ = config_.net_inp_height * config_.net_inp_width * config_.net_inp_channels;
        dst_ptr_d_ = reinterpret_cast<Npp8u *>(cudaFastMalloc(sizeof(Npp8u) * dst_pixel_num_ * config_.batch_size,
                                                              config_.device_id));
        if (!dst_ptr_d_)
        {
            AIALG_ASSERT(0);
        }

        dst_float_ptr_d_ = reinterpret_cast<Npp32f *>(cudaFastMalloc(
                sizeof(Npp32f) * dst_pixel_num_ * config_.batch_size, config_.device_id));
        if (!dst_float_ptr_d_)
        {
            AIALG_ASSERT(0);
        }

        dst_chw_float_ptr_d_ = reinterpret_cast<Npp32f *>(cudaFastMalloc(
                sizeof(Npp32f) * dst_pixel_num_ * config_.batch_size, config_.device_id));
        if (!dst_chw_float_ptr_d_)
        {
            AIALG_ASSERT(0);
        }

        // For batch gpu preprocess
        src_ptrs_d_.resize(config_.batch_size, nullptr);
        src_pixel_nums_pre_.resize(config_.batch_size, 0);
        coeffs_d_ = reinterpret_cast<Npp64f *>(cudaFastMalloc(sizeof(Npp64f) * 6 * config_.batch_size,
                                                              config_.device_id));
        if (!coeffs_d_)
        {
            AIALG_ASSERT(0);
        }
        pBatchList_d_ = reinterpret_cast<NppiWarpAffineBatchCXR *>(cudaFastMalloc(
                sizeof(NppiWarpAffineBatchCXR) * config_.batch_size, config_.device_id));
        if (!pBatchList_d_)
        {
            AIALG_ASSERT(0);
        }
        pBatchList_.resize(config_.batch_size);
    }


    void CModule_cls_tensorrt_impl::pre_process_gpu(const ImageInfoUint8 *imageInfoUint8, int bs)
    {
#ifdef AI_ALG_DEBUG
        std::cout << "Using pre_process_gpu" << std::endl;
#endif
        if (!imageInfoUint8->data)
        {
            CUDACHECK(cudaMemset(reinterpret_cast<Npp32f *>(buffers_->getDeviceBuffer(config_.input_names[0])) +
                                 bs * config_.net_inp_channels * config_.net_inp_width * config_.net_inp_height,
                                 0, sizeof(Npp32f) * config_.net_inp_channels * config_.net_inp_width *
                                    config_.net_inp_height));

            return;
        }
        /************Device memory allocator and initialization***********/
        size_t src_pixel_num = imageInfoUint8->img_height * imageInfoUint8->stride;
        if (src_pixel_nums_pre_[bs] < src_pixel_num)
        {
            cudaFastFree(src_ptrs_d_[bs], config_.device_id);
            src_pixel_nums_pre_[bs] = src_pixel_num;
            src_ptrs_d_[bs] = reinterpret_cast<Npp8u *>(cudaFastMalloc(sizeof(Npp8u) * src_pixel_nums_pre_[bs],
                                                                       config_.device_id));
        }
        if (imageInfoUint8->is_device_data)
        {
            CUDACHECK(cudaMemcpy(src_ptrs_d_[bs], imageInfoUint8->data, sizeof(Npp8u) * src_pixel_num,
                                 cudaMemcpyDeviceToDevice));
        } else
        {
            CUDACHECK(cudaMemcpy(src_ptrs_d_[bs], imageInfoUint8->data, sizeof(Npp8u) * src_pixel_num,
                                 cudaMemcpyHostToDevice));
        }

        /**********************getAffineTransform*************************/
        NppiRect oSrcROI = {.x=0, .y=0, .width=imageInfoUint8->img_width, .height=imageInfoUint8->img_height};
        double aQuad[4][2] = {{0.0,                         0.0},
                              {1.0 * config_.net_inp_width, 0.0},
                              {1.0 * config_.net_inp_width, 1.0 * config_.net_inp_height},
                              {0,                           1.0 * config_.net_inp_height}};
        double aCoeffs[2][3];
        nppiGetAffineTransform(oSrcROI, aQuad, aCoeffs);

        /**********************warpAffine**********************/
        nppiWarpAffine_8u_C3R(src_ptrs_d_[bs], {imageInfoUint8->img_width, imageInfoUint8->img_height},
                              sizeof(Npp8u) * imageInfoUint8->stride,
                              {0, 0, imageInfoUint8->img_width, imageInfoUint8->img_height},
                              dst_ptr_d_ + bs * dst_pixel_num_,
                              sizeof(Npp8u) * config_.net_inp_width * config_.net_inp_channels,
                              {0, 0, config_.net_inp_width, config_.net_inp_height},
                              aCoeffs,
                              NPPI_INTER_LINEAR);

//std::vector<uint8_t> img_after_warpAffine_data;
//img_after_warpAffine_data.resize(dst_pixel_num_);
//CUDACHECK(cudaMemcpy(img_after_warpAffine_data.data(), dst_ptr_d_,
//                     sizeof(Npp8u) * dst_pixel_num_, cudaMemcpyDeviceToHost));
//cv::Mat img_after_warpAffine = cv::Mat(config_.net_inp_height, config_.net_inp_width, CV_8UC3,
//                                       img_after_warpAffine_data.data());
//cv::imwrite("aaaaaaaaaaaa.jpg", img_after_warpAffine);

//        if (0 == config_.model_include_preprocess)
//        {
//            /**********************bgr2rgb*************************/
//            const int aDstOrder[3] = {2, 1, 0};
//            nppiSwapChannels_8u_C3IR(dst_ptr_d_ + bs * dst_pixel_num_,
//                                     sizeof(Npp8u) * config_.net_inp_width * config_.net_inp_channels,
//                                     {config_.net_inp_width, config_.net_inp_height},
//                                     aDstOrder);
//        }
        /**********************uint8 -> float*****************/
        nppiConvert_8u32f_C3R(dst_ptr_d_ + bs * dst_pixel_num_,
                              sizeof(Npp8u) * config_.net_inp_width * config_.net_inp_channels,
                              dst_float_ptr_d_ + bs * dst_pixel_num_,
                              sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels,
                              {config_.net_inp_width, config_.net_inp_height});

        Npp32f *net_input_buffer_d = reinterpret_cast<Npp32f *>(buffers_->getDeviceBuffer(config_.input_names[0]));
        if (0 == config_.model_include_preprocess)
        {
            /**********************(x - a) / b*********************/
            /*1.-------- y = (x - a) --------*/
            // const Npp32f means[3] = {0.0f, 0.0f, 0.0f};
            nppiSubC_32f_C3IR(config_.means, dst_float_ptr_d_ + bs * dst_pixel_num_,
                              sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels,
                              {config_.net_inp_width, config_.net_inp_height});

            /*2.---------- y * s ----------*/
            // const Npp32f scales[3] = {0.00392157f, 0.00392157f, 0.00392157f};
            nppiMulC_32f_C3IR(config_.scales, dst_float_ptr_d_ + bs * dst_pixel_num_,
                              sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels,
                              {config_.net_inp_width, config_.net_inp_height});

            /**********************hwc2chw*************************/
            Npp32f *const aDst[3] = {net_input_buffer_d + bs * dst_pixel_num_,
                                     net_input_buffer_d + bs * dst_pixel_num_ +
                                     config_.net_inp_width * config_.net_inp_height,
                                     net_input_buffer_d + bs * dst_pixel_num_ +
                                     2 * config_.net_inp_width * config_.net_inp_height};
            nppiCopy_32f_C3P3R(dst_float_ptr_d_ + bs * dst_pixel_num_,
                               sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels,
                               aDst,
                               sizeof(Npp32f) * config_.net_inp_width,
                               {config_.net_inp_width, config_.net_inp_height});
        } else
        {
            /*------copy preprocessed data to net input poiner------*/
            CUDACHECK(cudaMemcpy(
                    net_input_buffer_d + bs * dst_pixel_num_,
                    dst_float_ptr_d_ + bs * dst_pixel_num_,
                    sizeof(Npp32f) * dst_pixel_num_, cudaMemcpyDeviceToDevice));
        }
    }

    void CModule_cls_tensorrt_impl::pre_batch_process(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        if (1 == config_.model_include_preprocess)
        {
//        #pragma omp parallel for
            for (int bs = 0; bs < batch_size; ++bs)
            {
                if (!imageInfos[bs].data)
                {
                    continue;
                }
                int src_channels = 3;
                size_t src_pixel_num = imageInfos[bs].img_height * imageInfos[bs].img_width * src_channels;
                if (src_pixel_nums_pre_[bs] < src_pixel_num)
                {
                    cudaFastFree(src_ptrs_d_[bs], config_.device_id);
                    src_pixel_nums_pre_[bs] = src_pixel_num;
                    src_ptrs_d_[bs] = reinterpret_cast<Npp8u *>(cudaFastMalloc(sizeof(Npp8u) * src_pixel_nums_pre_[bs],
                                                                               config_.device_id));
                }
                if (imageInfos[bs].is_device_data)
                {
                    CUDACHECK(cudaMemcpy(src_ptrs_d_[bs], imageInfos[bs].data, sizeof(Npp8u) * src_pixel_num,
                                         cudaMemcpyDeviceToDevice));
                } else
                {
                    CUDACHECK(cudaMemcpy(src_ptrs_d_[bs], imageInfos[bs].data, sizeof(Npp8u) * src_pixel_num,
                                         cudaMemcpyHostToDevice));
                }

                pBatchList_[bs].pSrc = src_ptrs_d_[bs];
                pBatchList_[bs].nSrcStep = imageInfos[bs].stride;
                pBatchList_[bs].pDst = dst_ptr_d_ + bs * dst_pixel_num_;
                pBatchList_[bs].nDstStep = sizeof(Npp8u) * config_.net_inp_width * config_.net_inp_channels;
                /**********************getAffineTransform*************************/
                NppiRect oSrcROI = {.x=0, .y=0, .width=imageInfos[bs].img_width, .height=imageInfos[bs].img_height};
                double aQuad[4][2] = {{0.0,                         0.0},
                                      {1.0 * config_.net_inp_width, 0.0},
                                      {1.0 * config_.net_inp_width, 1.0 * config_.net_inp_height},
                                      {0.0,                         1.0 * config_.net_inp_height}};
                double aCoeffs[2][3];
                nppiGetAffineTransform(oSrcROI, aQuad, aCoeffs);

                CUDACHECK(cudaMemcpy(coeffs_d_ + bs * 6, aCoeffs, sizeof(Npp64f) * 6, cudaMemcpyHostToDevice));
                pBatchList_[bs].pCoeffs = coeffs_d_ + bs * 6;
            }
            CUDACHECK(cudaMemcpy(pBatchList_d_, pBatchList_.data(), sizeof(NppiWarpAffineBatchCXR) * config_.batch_size,
                                 cudaMemcpyHostToDevice));

            nppiWarpAffineBatchInit(pBatchList_d_, config_.batch_size);
            /**********************warpAffine**********************/
            nppiWarpAffineBatch_8u_C3R({imageInfos[0].img_width, imageInfos[0].img_height},
                                       {0, 0, imageInfos[0].img_width, imageInfos[0].img_height},
                                       {0, 0, config_.net_inp_height, config_.net_inp_width},
                                       NPPI_INTER_LINEAR,
                                       pBatchList_d_, config_.batch_size);

//        #pragma omp parallel for
            for (int bs = 0; bs < config_.batch_size; ++bs)
            {
                /**********************uint8 -> float*****************/
                nppiConvert_8u32f_C3R(dst_ptr_d_ + bs * dst_pixel_num_,
                                      sizeof(Npp8u) * config_.net_inp_width * config_.net_inp_channels,
                                      reinterpret_cast<Npp32f *>(buffers_->getDeviceBuffer(config_.input_names[0])) +
                                      bs * dst_pixel_num_,
                                      sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels,
                                      {config_.net_inp_width, config_.net_inp_height});
            }
        } else
        {
//        #pragma omp parallel for
            for (int bs = 0; bs < batch_size; ++bs)
            {
                pre_process_gpu(&imageInfos[bs], bs);
            }
        }
    }

    void CModule_cls_tensorrt_impl::engine_run()
    {
#ifdef AI_ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif
        bool status = context_->executeV2(buffers_->getDeviceBindings().data());
        if (!status)
        {
            AIALG_ERROR("Error %d line in file %s", __LINE__, __FILE__);
        }

#ifdef AI_ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
        AIALG_PRINT("TensorRT inference time %ld us\n",
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count());
        begin_time = std::chrono::system_clock::now();
#endif
        buffers_->copyOutputToHost();

        // get output
        int net_output_num = config_.output_names.size();
        // The numbers of output element;
        long out_data_num = 0;
        for (size_t idx = 0; idx < net_output_num; idx++)
        {
            auto mOutputDims = net_->getBindingDimensions(net_->getBindingIndex(config_.output_names[idx].c_str()));
//            auto mOutputDims = network_->getOutput(idx)->getDimensions();
            int num_of_elems = 1;
            for (int idy = 0; idy < mOutputDims.nbDims; ++idy)
            {
                num_of_elems *= mOutputDims.d[idy];
            }
            out_data_num += num_of_elems;
        }

        if (data_out_.size() < out_data_num)
        {
#ifdef AI_ALG_DEBUG
            std::cout << "Resize data_out_ : " << out_data_num << std::endl;
#endif
            data_out_.resize(out_data_num);
        }

        float *data_ = data_out_.data();
        for (size_t idx = 0; idx < net_output_num; idx++)
        {
            auto mOutputDims = net_->getBindingDimensions(net_->getBindingIndex(config_.output_names[idx].c_str()));
            int step = 1;
            for (int idy = 0; idy < mOutputDims.nbDims; ++idy)
            {
                step *= mOutputDims.d[idy];
            }
            float *output = static_cast<float*>(buffers_->getHostBuffer(config_.output_names[idx]));
            memcpy(data_, output, sizeof(float) * step);
        }
#ifdef AI_ALG_DEBUG
        end_time = std::chrono::system_clock::now();
        AIALG_PRINT("postprocess time %ld us\n",
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count());
#endif
    }

    bool CModule_cls_tensorrt_impl::constructNetwork(TRTUniquePtr<nvinfer1::IBuilder> &builder,
                                                     TRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                                     TRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                                     TRTUniquePtr<nvonnxparser::IParser> &parser)
    {

        auto parsed = parser->parseFromFile(config_.weights_path.c_str(),
                                            static_cast<int>(sample::gLogger.getReportableSeverity()));
        if (!parsed)
        {
            return false;
        }

        builder->setMaxBatchSize(config_.batch_size);
//    config->setMaxWorkspaceSize(2_GiB);
        if (config_.fp16)
        {
            if (isSupported(nvinfer1::DataType::kHALF))
            {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
        }
        if (config_.int8)
        {
            if (isSupported(nvinfer1::DataType::kINT8))
            {
                config->setFlag(nvinfer1::BuilderFlag::kINT8);
            }
#if NV_TENSORRT_MAJOR >= 8
            samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
#else
            samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
#endif
        }

        samplesCommon::enableDLA(builder.get(), config.get(), config_.dlaCore);

        return true;
    }

    bool CModule_cls_tensorrt_impl::isSupported(nvinfer1::DataType dataType)
    {
        auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder)
        {
            return false;
        }

        if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8())
            || (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16()))
        {
            return false;
        }

        return true;
    }

}