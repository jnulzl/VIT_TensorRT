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

static int open_video(const std::string& video_path, cv::Ptr<cv::cudacodec::VideoReader>& video_reader)
{
    video_reader = cv::cudacodec::createVideoReader(video_path);
    if (video_reader.empty())
    {
        std::cout << "Unable open video/camera " << video_path << std::endl;
        return -1;
    }

    cv::VideoCapture cap_tmp;
    cap_tmp.open(video_path);
    if (!cap_tmp.isOpened())
    {
        std::cout << "Unable open video/camera " << video_path << std::endl;
        return -1;
    }
    int fps = static_cast<int>(cap_tmp.get(cv::CAP_PROP_FPS));
    cap_tmp.release();
    return fps;
}

std::vector<std::string> split(const std::string& string, char separator, bool ignore_empty) {
    std::vector<std::string> pieces;
    std::stringstream ss(string);
    std::string item;
    while (getline(ss, item, separator)) {
        if (!ignore_empty || !item.empty()) {
            pieces.push_back(std::move(item));
        }
    }
    return pieces;
}

std::string trim(const std::string& str) {
    size_t left = str.find_first_not_of(' ');
    if (left == std::string::npos) {
        return str;
    }
    size_t right = str.find_last_not_of(' ');
    return str.substr(left, (right - left + 1));
}

int main(int argc, char* argv[])
{
    std::string project_root = std::string(PROJECT_ROOT);

    std::string weights_path;
    std::string deploy_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    float means_rgb[3] = {0.0f, 0.0f, 0.0f};
    float scales_rgb[3] = {1.0f, 1.0f, 1.0f};

    BaseConfig config_tmp;
    config_tmp.means[0] = means_rgb[0];
    config_tmp.means[1] = means_rgb[1];
    config_tmp.means[2] = means_rgb[2];
    config_tmp.scales[0] = scales_rgb[0];
    config_tmp.scales[1] = scales_rgb[1];
    config_tmp.scales[2] = scales_rgb[2];

    config_tmp.mean_length = 3;
    config_tmp.net_inp_channels = 3;

    if(argc < 3)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " model_path top_video_path"
                  << std::endl;
        return -1;
    }

    config_tmp.model_include_preprocess = 0;
    input_names.push_back("input1");
    output_names.push_back("output1");

    weights_path = argv[1];
    deploy_path = argv[1];
    config_tmp.net_inp_width = 224;
    config_tmp.net_inp_height = config_tmp.net_inp_width;
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = weights_path;
    config_tmp.deploy_path = deploy_path;

    config_tmp.batch_size = 1;
    config_tmp.device_id = 0;

    cv::cuda::setDevice(config_tmp.device_id);
    tensorrt_cls::CModule_cls cls;
    cls.init(config_tmp);

    std::vector<ImageInfoUint8> img_batch;
    img_batch.resize(config_tmp.batch_size);

    std::vector<cv::cuda::GpuMat> d_frames;
    d_frames.resize(config_tmp.batch_size);

    std::vector<cv::cuda::GpuMat> d_frame_bgrs;
    d_frame_bgrs.resize(config_tmp.batch_size);

    std::vector<std::string> video_paths;
    video_paths.emplace_back(argv[2]);

    cv::Ptr<cv::cudacodec::VideoReader> caps_reader;
    for (int idx = 0; idx < video_paths.size(); ++idx)
    {
        std::vector<std::string> video_names = split(video_paths[idx], '/', true);
        std::string video_name = video_names[video_names.size() - 1];
        caps_reader.release();
        int fps = open_video(video_paths[idx], caps_reader);
        if(fps < 0)
        {
            continue;
        }
        long frame_id = 0;
        std::vector<int> count_zhedang_ratio;
        count_zhedang_ratio.clear();
        bool begin_zhedang_flag = false;
        std::vector<long> begin_frame_ids;
        std::vector<float> begin_times;
        cv::Mat img_save;
        bool save_result_img = false;
        long save_frame_id = 0;
        float save_time = 0;
        while (true)
        {
            if (caps_reader->nextFrame(d_frames[0]))
            {
                cv::cuda::cvtColor(d_frames[0], d_frame_bgrs[0], cv::COLOR_BGRA2BGR);
                img_batch[0].data = d_frame_bgrs[0].data;
                img_batch[0].img_height = d_frame_bgrs[0].rows;
                img_batch[0].img_width = d_frame_bgrs[0].cols;
                img_batch[0].is_device_data = 1;
                img_batch[0].stride = d_frame_bgrs[0].step;
                img_batch[0].frame_id = frame_id;
                img_batch[0].img_data_type = InputDataType::IMG_BGR;
            }
            else
            {
                break;
            }
            std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
            cls.process_batch(img_batch.data(), img_batch.size());
            std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();

            bool is_zhedang = false;
            const ClsInfo* res = cls.get_result();
            /*****************************************check zhedang************************************/
            frame_id++;
            if(0 == res[0].label)
            {
                cv::Mat img_show;
                d_frame_bgrs[0].download(img_show);
                is_zhedang = true;
                begin_zhedang_flag = true;
                save_frame_id = frame_id;
                save_time = static_cast<float>(frame_id) / fps;
                if(save_result_img)
                {
                    img_save = img_show.clone();
                }
            }
            if(begin_zhedang_flag)
            {
                if(is_zhedang)
                {
                    count_zhedang_ratio.push_back(1);
                }
                else
                {
                    count_zhedang_ratio.push_back(0);
                }
            }

            if(count_zhedang_ratio.size() >= 60)
            {
                float sum = 0.0f;
                for (int idy = 0; idy < count_zhedang_ratio.size(); ++idy)
                {
                    sum += count_zhedang_ratio[idy];
                }
                float zhedang_ratio = sum / (count_zhedang_ratio.size() + 0.01f);
                if(zhedang_ratio > 0.5)
                {
                    if(save_result_img)
                    {
                        cv::imwrite("./res/" + std::to_string(save_frame_id) + "_zhedang.jpg", img_save);
                    }
                    begin_frame_ids.push_back(save_frame_id);
                    begin_times.push_back(save_time);
                    std::printf("fps = %d, is_zhedang = 1, frame_id = %ld, time = %f\n", fps, save_frame_id, static_cast<double>(save_frame_id) / fps);
                }
                count_zhedang_ratio.clear();
                begin_zhedang_flag = false;
            }
        }

        std::string video_name_path = video_name.substr(0, video_name.size() - 4) + ".json";
        FILE* fpW = fopen(video_name_path.c_str(), "w");
        fprintf(fpW, "{\n");
        fprintf(fpW, "\t\"is_offset\" : 0,\n");
        fprintf(fpW, "\t\"items\" : [\n");
        if(1 == begin_frame_ids.size())
        {
            fprintf(fpW, "\t\t{\n");
            fprintf(fpW, "\t\t\t\"frame_id\" : %ld,\n", begin_frame_ids[0]);
            fprintf(fpW, "\t\t\t\"time_second\" : %f\n", begin_times[0]);
            fprintf(fpW, "\t\t}\n");
        }
        else if(begin_frame_ids.size() > 1)
        {
            int item_num = begin_frame_ids.size() - 1;
            for (int idy = 0; idy < item_num; ++idy)
            {
                fprintf(fpW, "\t\t{\n");
                fprintf(fpW, "\t\t\t\"frame_id\" : %ld,\n", begin_frame_ids[idy]);
                fprintf(fpW, "\t\t\t\"time_second\" : %f\n", begin_times[idy]);
                fprintf(fpW, "\t\t},\n");
            }

            fprintf(fpW, "\t\t{\n");
            fprintf(fpW, "\t\t\t\"frame_id\" : %ld,\n", begin_frame_ids[item_num]);
            fprintf(fpW, "\t\t\t\"time_second\" : %f\n", begin_times[item_num]);
            fprintf(fpW, "\t\t}\n");
        }

        fprintf(fpW, "\t]\n");
        fprintf(fpW, "}\n");
        fclose(fpW);
    }
    cls.deinit();
    return 0;
}
