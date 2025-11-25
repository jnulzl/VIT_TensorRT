//
// Created by lizhaoliang-os on 2020/6/23.

#include <iostream>
#include <string>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"

#include "vit/Module_vit.h"
#include "utils/file_process.hpp"


int main(int argc, char* argv[])
{
    std::string project_root = std::string(PROJECT_ROOT);

    std::string weights_path;
    std::string deploy_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    input_names.push_back("pixel_values");
    output_names.push_back("vit_embeds");

    std::string input_src;

    BaseConfig config_tmp;
    float means_rgb[3] = { 123.675f, 116.28f , 103.53f };
    float scales_rgb[3] = { 0.0171f, 0.0175f  , 0.0174f };

    config_tmp.means[0] = means_rgb[0];
    config_tmp.means[1] = means_rgb[1];
    config_tmp.means[2] = means_rgb[2];
    config_tmp.scales[0] = scales_rgb[0];
    config_tmp.scales[1] = scales_rgb[1];
    config_tmp.scales[2] = scales_rgb[2];

    config_tmp.mean_length = 3;
    config_tmp.net_inp_channels = 3;
    config_tmp.model_include_preprocess = 0;

    if (argc < 7)
    {
        std::cout << "Usage:\n\t "
            << argv[0] << " trt_model_path input_size batch_size device_id is_save_res image_list_txt"
            << std::endl;
        return -1;
    }
    std::vector<std::string> image_list;
    alg_utils::get_all_line_from_txt(argv[6], image_list);

    weights_path = std::string(argv[1]);
    config_tmp.net_inp_width = std::atoi(argv[2]);
    config_tmp.net_inp_height = config_tmp.net_inp_width;
    config_tmp.batch_size = std::atoi(argv[3]);
    config_tmp.device_id = std::atoi(argv[4]);
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = weights_path;
    config_tmp.deploy_path = weights_path;

    tensorrt_vit::CModule_vit vit;
    std::cout << "Loading trt model from " << weights_path << std::endl;
    vit.init(config_tmp);
    std::cout << "Loading trt model end!" << std::endl;

    int is_save_res = std::atoi(argv[5]);

    cv::cuda::setDevice(config_tmp.device_id);
    long frame_id = 0;
    std::vector<ImageInfoUint8> img_batch(config_tmp.batch_size);
    std::vector<cv::Mat> frame_bgrs(config_tmp.batch_size);
    int batch_num = image_list.size() / config_tmp.batch_size;
    for (int idx = 0; idx < batch_num; idx++)
    {
        for (int bs = 0; bs < config_tmp.batch_size; ++bs)
        {
            frame_bgrs[bs] = cv::imread(image_list[idx * config_tmp.batch_size + bs]);
            img_batch[bs].data = frame_bgrs[bs].data;
            img_batch[bs].img_height = frame_bgrs[bs].rows;
            img_batch[bs].img_width = frame_bgrs[bs].cols;
            img_batch[bs].is_device_data = 0;
            img_batch[bs].stride = frame_bgrs[bs].step;
            img_batch[bs].frame_id = frame_id;
            img_batch[bs].img_data_type = InputDataType::IMG_BGR;
        }

        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        vit.process_batch(img_batch.data(), config_tmp.batch_size);
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();

        const NetFloatTensor* res = vit.get_result();
        std::cout << "Frame = " << frame_id << " Batch = " << config_tmp.batch_size << " TensorRT process time = " << std::chrono::duration_cast<std::chrono::microseconds>(finishTP1 - startTP).count() << " us" << std::endl;
        if (is_save_res)
        {
            int vit_embedding_size = res->channels * res->height;
            for (int bs = 0; bs < config_tmp.batch_size; ++bs)
            {
                if (res->data)
                {
                    for (int idx = 0; idx < vit_embedding_size; idx++)
                    {
                        std::printf("%.6f ", res->data[idx + bs * vit_embedding_size]);
                        if (0 == (idx + 1) % res->height)
                        {
                            std::printf("\n");
                        }
                    }
                }
                else
                {
                    std::cout << "Video is end!" << std::endl;
                }
            }
        }

        frame_id++;
    }

    vit.deinit();
    return 0;
}
