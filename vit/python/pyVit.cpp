//
// Created by Li Zhaoliang on 2024/11/13.
//
//
// Created by lizhaoliang-os on 2021/1/28.
//

#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>

#include <torch/torch.h>
#include <cuda_runtime.h>

#include "Module_vit.h"
#include "debug.h"

namespace py = pybind11;

class PyVit
{
public:
    PyVit() = default;

    ~PyVit() = default;

    void init(const BaseConfig& config)
    {
        batch_size_ = config.batch_size;
        obj_.init(config);
    }

    void deinit()
    {
        obj_.deinit();
    }

    void process(const std::vector<py::array_t<uint8_t>>& imgs)
    {
        int num_img = imgs.size();
        if (num_img != batch_size_)
        {
            std::printf("Input image number %d must be equal to batch size %d!\n", num_img, batch_size_);
            return;
        }

        std::vector<py::buffer_info> img_bufs(num_img);
        std::vector<ImageInfoUint8> ImageInfoUint8s(num_img);
        for (int idx = 0; idx < num_img; ++idx)
        {
            const py::array_t<uint8_t>& img = imgs[idx];
            if (!img.request().ptr)
            {
                std::printf("Input image[%d] is empty!\n", idx);
                ImageInfoUint8s[idx].data = nullptr;
                ImageInfoUint8s[idx].img_height = 0;
                ImageInfoUint8s[idx].img_width = 0;
                ImageInfoUint8s[idx].is_device_data = 0;
                ImageInfoUint8s[idx].stride = 0;
                ImageInfoUint8s[idx].frame_id = 0;
                ImageInfoUint8s[idx].img_data_type = InputDataType::IMG_BGR;
                continue;
            }
            // Check if the input image is a valid 3-channel BGR image
            img_bufs[idx] = img.request();
            if (3 != img_bufs[idx].ndim || 3 != img_bufs[idx].shape[2])
            {
                std::printf("Input image[%d] must have three channels!\n", idx);
            }

            int src_height = static_cast<int>(img_bufs[idx].shape[0]);
            int src_width = static_cast<int>(img_bufs[idx].shape[1]);

#ifdef AI_ALG_DEBUG
            for (int idx = 0; idx < img.ndim(); ++idx)
            {
                std::cout << "Shape : " << idx << " = " << img.shape(idx) << std::endl;
            }
#endif
            ImageInfoUint8s[idx].data = reinterpret_cast<uint8_t*>(img_bufs[idx].ptr);
            ImageInfoUint8s[idx].img_height = src_height;
            ImageInfoUint8s[idx].img_width = src_width;
            ImageInfoUint8s[idx].is_device_data = 0;
            ImageInfoUint8s[idx].stride = src_width * img_bufs[idx].ndim;
            ImageInfoUint8s[idx].frame_id = 0;
            ImageInfoUint8s[idx].img_data_type = InputDataType::IMG_BGR;
        }

        obj_.process_batch(ImageInfoUint8s.data(), num_img);
    }

    torch::Tensor get_result()
    {
        const NetFloatTensor* res = obj_.get_result();
        float* dptr = res->data;
        int N = res->batch;
        int C = res->channels;
        int H = res->height;
        auto options = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA);

        // 不释放 GPU pointer，因为释放由C++库管理
        return torch::from_blob(
            dptr,
            { N, C, H },
            [](void* p) { /* no free */ },
            options
        );
    }

private:
    tensorrt_vit::CModule_vit obj_;
    int batch_size_ = 0;
};

PYBIND11_MODULE(MODEL_NAME, m)
{

    m.doc() = "Objdet Python Wrapper";

    //wrapper C++ YoloConfig to python YoloConfig
    py::class_<BaseConfig>(m, "BaseConfig", py::module_local())
        .def(py::init())
        .def_readwrite("input_names", &BaseConfig::input_names)
        .def_readwrite("output_names", &BaseConfig::output_names)
        .def_readwrite("weights_path", &BaseConfig::weights_path)
        .def_readwrite("deploy_path", &BaseConfig::deploy_path)
        .def_property_readonly("means", [](py::object& obj) {
        const BaseConfig& o = obj.cast<BaseConfig&>();
        return py::array{ 3, o.means, obj };
            }
        )
        .def_property_readonly("scales", [](py::object& obj) {
        const BaseConfig& o = obj.cast<BaseConfig&>();
        return py::array{ 3, o.scales, obj };
            }
        )
        .def_readwrite("mean_length", &BaseConfig::mean_length)
        .def_readwrite("net_inp_channels", &BaseConfig::net_inp_channels)
        .def_readwrite("net_inp_width", &BaseConfig::net_inp_width)
        .def_readwrite("net_inp_height", &BaseConfig::net_inp_height)
        .def_readwrite("num_threads", &BaseConfig::num_threads)
#ifdef USE_CUDA
        .def_readwrite("batch_size", &BaseConfig::batch_size)
        .def_readwrite("device_id", &BaseConfig::device_id)
#ifdef USE_TENSORRT
        .def_readwrite("dlaCore", &BaseConfig::dlaCore)
        .def_readwrite("fp16", &BaseConfig::fp16)
        .def_readwrite("int8", &BaseConfig::int8)
#endif // USE_CUDA
#endif // USE_TENSORRT
        .def_readwrite("model_include_preprocess", &BaseConfig::model_include_preprocess)
        ;

    // //wrapper C++ YoloConfig to python YoloConfig
    // py::class_<YoloConfig, BaseConfig>(m, "YoloConfig")
    //         .def(py::init())
    //         .def_readwrite("num_cls", &YoloConfig::num_cls)
    //         .def_readwrite("conf_thres", &YoloConfig::conf_thres)
    //         .def_readwrite("nms_thresh", &YoloConfig::nms_thresh)
    //         .def_readwrite("strides", &YoloConfig::strides)
    //         .def_readwrite("anchor_grids", &YoloConfig::anchor_grids);

    // //wrapper C++ BoxInfo to python BoxInfo
    // py::class_<BoxInfo>(m, "BoxInfo")
    //         .def(py::init())
    //         .def_readwrite("x1", &BoxInfo::x1)
    //         .def_readwrite("y1", &BoxInfo::y1)
    //         .def_readwrite("x2", &BoxInfo::x2)
    //         .def_readwrite("y2", &BoxInfo::y2)
    //         .def_readwrite("score", &BoxInfo::score)
    //         .def_readwrite("area", &BoxInfo::area)
    //         .def_readwrite("label", &BoxInfo::label);

    py::class_<PyVit>(m, "PyVit")
        .def(py::init())
        .def("init", &PyVit::init)
        .def("deinit", &PyVit::deinit)
        .def("process", &PyVit::process)
        .def("get_result", &PyVit::get_result);

}
