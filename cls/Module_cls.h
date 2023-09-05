#ifndef MODULE_CLS_H
#define MODULE_CLS_H

#include <string>
#include <vector>
#include "data_type.h"
#include "alg_define.h"

namespace tensorrt_cls
{
    class AIALG_PUBLIC CModule_cls
    {
    public:
        CModule_cls();

        ~CModule_cls();

        void init(const BaseConfig &config);

        void deinit();

        void process_batch(const ImageInfoUint8 *imageInfos, int batch_size);

        const ClsInfo *get_result();

    private:
        AW_ANY_POINTER impl_;
    };
}

#endif // MODULE_CLS_H

