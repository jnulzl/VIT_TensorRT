#include <assert.h>
#include <string.h>

#include "Module_cls_impl.h"
#include "alg_define.h"
#include "debug.h"

static void softmax(float* vec, size_t len)
{
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++)
    {
        sum += std::exp(vec[i]);
    }
    for (size_t i = 0; i < len; i++)
    {
        vec[i] = exp(vec[i]) / sum;
    }
}

namespace tensorrt_cls
{
    CModule_cls_impl::CModule_cls_impl()
    {

    }

    CModule_cls_impl::~CModule_cls_impl()
    {

    }

    void CModule_cls_impl::deinit()
    {
        engine_deinit();
#ifdef AI_ALG_DEBUG
        AIALG_PRINT("release success!\n");
#endif
    }

    void CModule_cls_impl::init(const BaseConfig &config)
    {
        config_ = config;
        engine_init();

        frame_ids_.resize(config_.batch_size);
        cls_batch_.resize(config_.batch_size);
    }

    void CModule_cls_impl::pre_batch_process(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        //pre_batch_process cpu version
    }


    void CModule_cls_impl::process_batch(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        //TODO : check if batch_size == config_.batch_size
        for (int bs = 0; bs < config_.batch_size; ++bs)
        {
            frame_ids_[bs] = imageInfos[bs].frame_id;
        }

#ifdef AI_ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif

        pre_batch_process(imageInfos, batch_size);

#ifdef AI_ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
        AIALG_PRINT("Preprocess time %ld us\n",
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count());
#endif

        engine_run();

#ifdef AI_ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time_run = std::chrono::system_clock::now();
        AIALG_PRINT("Inference time %ld us\n",
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time_run - end_time).count());
#endif

        post_process();

#ifdef AI_ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time_post = std::chrono::system_clock::now();
        AIALG_PRINT("Postprocess time %ld us\n",
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time_post - end_time_run).count());
#endif
    }

    void CModule_cls_impl::post_process()
    {
        int num_cls = data_out_.size() / config_.batch_size;
        for (int bs = 0; bs < config_.batch_size; ++bs)
        {
            float* score_bs = data_out_.data() + bs * num_cls;
            softmax(score_bs, num_cls);
            int max_index = 0;
            float max_score = score_bs[0];
            for (int idx = 1; idx < num_cls; ++idx)
            {
                if(score_bs[idx] > max_score)
                {
                    max_score = score_bs[idx];
                    max_index = idx;
                }
            }
            cls_batch_[bs].label = max_index;
            cls_batch_[bs].score = max_score;
            cls_batch_[bs].frame_id = frame_ids_[bs];
        }
    }

    const ClsInfo* CModule_cls_impl::get_result()
    {
        return cls_batch_.data();
    }

    const BaseConfig* CModule_cls_impl::get_config() const
    {
        return &config_;;
    }
}
