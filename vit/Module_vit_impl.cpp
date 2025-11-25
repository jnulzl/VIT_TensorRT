#include <assert.h>
#include <string.h>

#include "Module_vit_impl.h"
#include "alg_define.h"
#include "debug.h"

namespace tensorrt_vit
{
    CModule_vit_impl::CModule_vit_impl() = default;

    CModule_vit_impl::~CModule_vit_impl() = default;

    void CModule_vit_impl::deinit()
    {
        engine_deinit();
#ifdef AI_ALG_DEBUG
        AIALG_PRINT("release success!\n");
#endif
    }

    void CModule_vit_impl::init(const BaseConfig& config)
    {
        config_ = config;
        engine_init();

        frame_ids_.resize(config_.batch_size);
    }

    void CModule_vit_impl::pre_batch_process(const ImageInfoUint8* imageInfos, int batch_size)
    {
        //pre_batch_process cpu version
    }


    void CModule_vit_impl::process_batch(const ImageInfoUint8* imageInfos, int batch_size)
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

    void CModule_vit_impl::post_process()
    {

    }

    const NetFloatTensor* CModule_vit_impl::get_result() const
    {
        return &data_out_gpu_tensor_;
    }

    const BaseConfig* CModule_vit_impl::get_config() const
    {
        return &config_;;
    }
}
