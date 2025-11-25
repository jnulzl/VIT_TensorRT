#ifndef MODULE_VIT_IMPL_H
#define MODULE_VIT_IMPL_H

#include <string>
#include <vector>

#include "data_type.h"

namespace tensorrt_vit
{
    class CModule_vit_impl
    {
    public:
        CModule_vit_impl();

        virtual ~CModule_vit_impl();

        void init(const BaseConfig& config);

        void deinit();

        void process_batch(const ImageInfoUint8* imageInfos, int batch_size);

        const NetFloatTensor* get_result() const;

        const BaseConfig* get_config() const;

    protected:
        virtual void pre_batch_process(const ImageInfoUint8* imageInfos, int batch_size);

        virtual void post_process();

        virtual void engine_init() = 0;

        virtual void engine_run() = 0;

        virtual void engine_deinit() = 0;

    protected:
        BaseConfig config_;
        NetFloatTensor data_out_gpu_tensor_{};

        std::vector<float> data_out_;
        std::vector<int> frame_ids_;
    };
}
#endif // MODULE_VIT_IMPL_H

