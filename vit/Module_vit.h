#ifndef MODULE_VIT_H
#define MODULE_VIT_H

#include <string>
#include <vector>
#include "data_type.h"
#include "alg_define.h"

namespace tensorrt_vit
{
    class AIALG_PUBLIC CModule_vit
    {
    public:
        CModule_vit();

        ~CModule_vit();

        void init(const BaseConfig &config);

        void deinit();

        void process_batch(const ImageInfoUint8 *imageInfos, int batch_size);

        const SegmentResult* get_result();

        const BaseConfig* get_config() const;

    private:
        AW_ANY_POINTER impl_;
    };
}

#endif // MODULE_VIT_H

