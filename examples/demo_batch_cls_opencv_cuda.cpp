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

#include "cls/Module_cls.h"
#include "utils/file_process.hpp"

void open_video(const std::string& video_path, cv::Ptr<cv::cudacodec::VideoReader>& video_reader)
{
    video_reader = cv::cudacodec::createVideoReader(video_path);
    if (video_reader.empty())
    {
        std::cout << "Unable open video/camera " << video_path << std::endl;
        return;
    }
}


int main(int argc, char* argv[])
{
    std::string project_root = std::string(PROJECT_ROOT);

    std::string weights_path;
    std::string deploy_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    input_names.push_back("input1");
    output_names.push_back("output1");

    std::string input_src;

    BaseConfig config_tmp;
    float means_rgb[3] = {0.0f, 0.0f, 0.0f};
    float scales_rgb[3] = {1.0f, 1.0f, 1.0f};

    config_tmp.means[0] = means_rgb[0];
    config_tmp.means[1] = means_rgb[1];
    config_tmp.means[2] = means_rgb[2];
    config_tmp.scales[0] = scales_rgb[0];
    config_tmp.scales[1] = scales_rgb[1];
    config_tmp.scales[2] = scales_rgb[2];

    config_tmp.mean_length = 3;
    config_tmp.net_inp_channels = 3;
    config_tmp.model_include_preprocess = 0;

    if(argc < 7)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " trt_model_path input_size batch_size device_id is_save_res video_prefix_list_txt"
                  << std::endl;
        return -1;
    }
    std::vector<std::string> video_names_prefix;
    alg_utils::get_all_line_from_txt(argv[6], video_names_prefix);

    weights_path = std::string(argv[1]);
    config_tmp.net_inp_width = std::atoi(argv[2]);
    config_tmp.net_inp_height = config_tmp.net_inp_width;
    config_tmp.batch_size = std::atoi(argv[3]);
    config_tmp.device_id = std::atoi(argv[4]);
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = weights_path;
    config_tmp.deploy_path = weights_path;

    tensorrt_cls::CModule_cls cls;
    std::cout << "Loading trt model from " << weights_path << std::endl;
    cls.init(config_tmp);
    std::cout << "Loading trt model end!" << std::endl;

    int is_save_res = std::atoi(argv[5]);

    cv::cuda::setDevice(config_tmp.device_id);
    std::vector<std::string> top_videos;
    top_videos.resize(config_tmp.batch_size);
    for (int bs = 0; bs < config_tmp.batch_size; ++bs)
    {
        top_videos[bs] = video_names_prefix[bs] + "_top.mp4";
    }
    std::vector<cv::Ptr<cv::cudacodec::VideoReader>> caps;
    caps.resize(config_tmp.batch_size);
    for (int bs = 0; bs < config_tmp.batch_size; ++bs)
    {
        std::cout << "Open video: " << top_videos[bs] << std::endl;
        open_video(top_videos[bs], caps[bs]);
    }

    long frame_id = 0;
    std::vector<ImageInfoUint8> img_batch;
    img_batch.resize(config_tmp.batch_size);

    std::vector<cv::cuda::GpuMat> d_frames;
    d_frames.resize(config_tmp.batch_size);

    std::vector<cv::cuda::GpuMat> d_frame_bgrs;
    d_frame_bgrs.resize(config_tmp.batch_size);
    while (true)
    {
        int count_empty_frame = 0;
        for (int bs = 0; bs < config_tmp.batch_size; ++bs)
        {
            if (caps[bs]->nextFrame(d_frames[bs]))
            {
                //cv::Mat img;
                //d_frames[bs].download(img);
                //cv::imwrite("./res/" + std::to_string(frame_id) + ".jpg", img);
                cv::cuda::cvtColor(d_frames[bs], d_frame_bgrs[bs], cv::COLOR_BGRA2BGR);
                img_batch[bs].data = d_frame_bgrs[bs].data;
                img_batch[bs].img_height = d_frame_bgrs[bs].rows;
                img_batch[bs].img_width = d_frame_bgrs[bs].cols;
                img_batch[bs].is_device_data = 1;
                img_batch[bs].stride = d_frame_bgrs[bs].step;
                img_batch[bs].frame_id = frame_id;
                img_batch[bs].img_data_type = InputDataType::IMG_BGR;
            }
            else
            {
                img_batch[bs].data = nullptr;
                count_empty_frame++;
            }
        }
        if(config_tmp.batch_size == count_empty_frame)
        {
            break;
        }

        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        cls.process_batch(img_batch.data(), config_tmp.batch_size);
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();

        const ClsInfo* res = cls.get_result();
        std::cout << "Frame = " << frame_id << " Batch = " << config_tmp.batch_size << " TensorRT process time = " << std::chrono::duration_cast<std::chrono::microseconds>(finishTP1 - startTP).count() << " us" << std::endl;
        for (int bs = 0; bs < config_tmp.batch_size; ++bs)
        {
            if(img_batch[bs].data)
            {
                std::cout << "Video " << bs << " label : " << res[bs].label << " score : " << res[bs].label << std::endl;
            }
            else
            {
                std::cout << "Video is end!" << std::endl;
            }
        }

        if(1 == is_save_res)
        {
            //show result
            for (int bs = 0; bs < config_tmp.batch_size; ++bs)
            {
                if(img_batch[bs].data)
                {
                    cv::Mat img_show;
                    d_frame_bgrs[bs].download(img_show);
                    int xmin    = 50;
                    int ymin    = 50;
                    int xmax    = img_batch[bs].img_width - 50;
                    int ymax    = img_batch[bs].img_height - 50;
                    float score = res[bs].score;
                    int label   = res[bs].label;
                    std::cout << "xyxy : " << xmin << " " << ymin << " " << xmax << " " << ymax << " " << score << " " << label << std::endl;
                    cv::putText(img_show, std::to_string(label), cv::Point(xmin, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255), 2);
                    cv::putText(img_show, std::to_string(score), cv::Point(xmax, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
                }
            }
        }

        frame_id++;
    }

    for (int bs = 0; bs < config_tmp.batch_size; ++bs)
    {
        caps[bs].release();
    }

    cls.deinit();
    return 0;
}
