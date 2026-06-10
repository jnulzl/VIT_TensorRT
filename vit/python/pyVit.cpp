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
#include <c10/cuda/CUDAStream.h>
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
        device_id_ = config.device_id;
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

    void process_with_tensor(torch::Tensor imgs)
    {
        cudaSetDevice(device_id_);

        imgs = imgs.clone().contiguous(); // Ensure the tensor is contiguous and avoid potential issues with non-contiguous memory
        TORCH_CHECK(imgs.is_contiguous(), "tensor must be contiguous");
        TORCH_CHECK(imgs.dtype() == torch::kUInt8, "expect uint8");
        TORCH_CHECK(4 == imgs.dim(), "Input tensor must have 4 dimensions!");
        const auto& imgs_shape = imgs.sizes();
        TORCH_CHECK(3 == imgs_shape[3], "Input tensor must have 3 channels and NHWC format!");

        int num_img = imgs_shape[0];
        int img_height = imgs_shape[1];
        int img_width = imgs_shape[2];
        int img_channels = imgs_shape[3];
        std::vector<ImageInfoUint8> ImageInfoUint8s(num_img);
        uint8_t* ptr = imgs.data_ptr<uint8_t>();
        int img_stride = imgs.numel() / num_img;
        for(int idx = 0; idx < num_img; ++idx)
        {
            ImageInfoUint8s[idx].data = ptr + idx * img_stride;
            ImageInfoUint8s[idx].img_height = img_height;
            ImageInfoUint8s[idx].img_width = img_width;
            ImageInfoUint8s[idx].is_device_data = imgs.is_cuda() ? 1 : 0;
            ImageInfoUint8s[idx].stride = img_width * img_channels;
            ImageInfoUint8s[idx].frame_id = 0;
            ImageInfoUint8s[idx].img_data_type = InputDataType::IMG_BGR;
        }

        obj_.process_batch(ImageInfoUint8s.data(), num_img);
    }

    void process_with_pretensor(torch::Tensor tensor_nchw)
    {
        cudaSetDevice(device_id_);
        tensor_nchw = tensor_nchw.clone().contiguous(); // Ensure the tensor is contiguous and avoid potential issues with non-contiguous memory
        TORCH_CHECK(tensor_nchw.is_contiguous(), "tensor must be contiguous");
        TORCH_CHECK(tensor_nchw.dtype() == torch::kFloat, "expect float32");
        TORCH_CHECK(4 == tensor_nchw.dim(), "Input tensor must have 4 dimensions!");
        const auto& img_shape = tensor_nchw.sizes();

        int num_img = img_shape[0];
        int img_channels = img_shape[1];
        int img_height = img_shape[2];
        int img_width = img_shape[3];

        NetFloatTensor floatTensor;
        floatTensor.data = tensor_nchw.data_ptr<float>();
        floatTensor.format = NetFloatTensor::DimensionType::NCHW;
        floatTensor.batch = num_img;
        floatTensor.channels = img_channels;
        floatTensor.height = img_height;
        floatTensor.width = img_width;

        obj_.process_batch(&floatTensor);
    }

    std::vector<torch::Tensor> get_result()
    {
        cudaSetDevice(device_id_);
        torch::NoGradGuard no_grad; // 关闭状态与梯度追踪

        const NetFloatTensor* res = obj_.get_result();
        std::vector<torch::Tensor> output_tensors;

        auto options = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA);

        for (size_t idx = 0; idx < obj_.get_config()->output_names.size(); ++idx)
        {
            const NetFloatTensor& tensor = res[idx];
            std::vector<int64_t> sizes = { tensor.batch, tensor.channels, tensor.height };

            // 1. 显式在 GPU 上直接分配一块属于 PyTorch 原生的干净连续显存
            // 通过这种方式创建出来的 Tensor 具备一切合法的元数据修改权限，后续绝不会触发 set_stride 限制
            torch::Tensor output_tensor = torch::empty(sizes, options);

            // 2. 拿到 PyTorch 新分配的 GPU 目标显存指针
            float* dst_ptr = output_tensor.data_ptr<float>();

            // 3. 计算需要拷贝的数据大小（字节数）
            size_t bytes = output_tensor.numel() * sizeof(float);

            // 4. 获取 PyTorch 当前正在使用的 CUDA 推理流
            // 这一步在最新版 PyTorch 中位于 c10::cuda 命名空间下
            cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream().stream();

            // 5. 异步将 TensorRT 的结果数据拷贝至 PyTorch 自建的显存中
            // 采用 DeviceToDevice 异步拷贝，既安全又保证了与原本方案一模一样的极致性能
            cudaMemcpyAsync(dst_ptr, 
                            tensor.data, 
                            bytes, 
                            cudaMemcpyDeviceToDevice, 
                            current_stream);

            output_tensors.push_back(output_tensor);
        }
        return output_tensors;
    }

private:
    tensorrt_vit::CModule_vit obj_;
    int batch_size_ = 0;
    int device_id_ = 0;
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

    py::class_<PyVit>(m, "PyVit")
        .def(py::init())
        .def("init", &PyVit::init)
        .def("deinit", &PyVit::deinit)
        .def("process", &PyVit::process)
        .def("process_with_tensor", &PyVit::process_with_tensor)
        .def("process_with_pretensor", &PyVit::process_with_pretensor)
        .def("get_result", &PyVit::get_result);

}
