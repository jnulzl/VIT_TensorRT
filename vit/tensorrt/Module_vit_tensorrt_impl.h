//
// Created by lizhaoliang-os on 2020/6/9.
//

#ifndef MODULE_VIT_TENSORRT_IMPL_H
#define MODULE_VIT_TENSORRT_IMPL_H

#include <memory>

#include <NvInferRuntime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <buffers.h>
#include <common.h> // for include samplesCommon::InferDeleter
#include <npp.h> // For gpu preprocess function(affineTransform, bgr2rgb, chw, (x - a) * b)

#include "Module_vit_impl.h"

namespace tensorrt_vit
{
    class CModule_vit_tensorrt_impl : public CModule_vit_impl
    {
    public:
        CModule_vit_tensorrt_impl();

        virtual ~CModule_vit_tensorrt_impl();

    private:
        virtual void engine_init() override;

        virtual void engine_run() override;

        virtual void engine_deinit() override;

        virtual void pre_batch_process(const ImageInfoUint8 *imageInfos, int batch_size) override;

        void pre_process_gpu(const ImageInfoUint8 *imageInfoUint8, int batch_id = 0);

    private:

        template<typename T>
        using TRTUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

        //!
        //! \brief Parses an ONNX model and creates a TensorRT network
        //!
        bool constructNetwork(TRTUniquePtr<nvinfer1::IBuilder> &builder,
                              TRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                              TRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                              TRTUniquePtr<nvonnxparser::IParser> &parser);

        bool isSupported(nvinfer1::DataType dataType);

    private:
        TRTUniquePtr<nvinfer1::IBuilder> builder_;
        TRTUniquePtr<nvinfer1::INetworkDefinition> network_;        
        TRTUniquePtr<nvinfer1::IBuilderConfig> config_trt_;
        TRTUniquePtr<nvonnxparser::IParser> parser_;

        TRTUniquePtr<nvinfer1::IRuntime> runtime_;    //!< The TensorRT runtime used to deserialize the engine
        std::shared_ptr<nvinfer1::ICudaEngine> net_; //!< The TensorRT engine used to run the network

        // Create RAII buffer manager object
        std::unique_ptr<samplesCommon::BufferManager> buffers_;
        TRTUniquePtr<nvinfer1::IExecutionContext> context_;

        // For gpu preprocess
        size_t dst_pixel_num_;
        Npp8u *dst_ptr_d_;
        Npp32f *dst_float_ptr_d_;
        Npp32f *dst_chw_float_ptr_d_;

        // For batch gpu preprocess
        std::vector<Npp8u *> src_ptrs_d_;
        std::vector<size_t> src_pixel_nums_pre_;
        Npp64f *coeffs_d_;
        NppiWarpAffineBatchCXR *pBatchList_d_;
        std::vector<NppiWarpAffineBatchCXR> pBatchList_;
    };
}
#endif //MODULE_CLS_TENSORRT_IMPL_H
