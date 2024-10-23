#include "Module_cls_impl.h"
#include "Module_cls.h"
#include "ai_alg_version.h"

#include "alg_define.h"
#include "debug.h"

namespace tensorrt_cls
{
    CModule_cls::~CModule_cls()
    {
        if(ANY_POINTER_CAST(impl_, CModule_cls_impl))
        {
            delete ANY_POINTER_CAST(impl_, CModule_cls_impl);
        }
    }

    void CModule_cls::init(const BaseConfig &config)
    {
        print_version();
        ANY_POINTER_CAST(impl_, CModule_cls_impl)->init(config);
    }

    void CModule_cls::deinit()
    {
        ANY_POINTER_CAST(impl_, CModule_cls_impl)->deinit();
#if defined(ALG_DEBUG) || defined(AI_ALG_DEBUG)
        AIALG_PRINT("release success!\n");
#endif
    }

    void CModule_cls::process_batch(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        ANY_POINTER_CAST(impl_, CModule_cls_impl)->process_batch(imageInfos, batch_size);
    }

    const ClsInfo *CModule_cls::get_result()
    {
        return ANY_POINTER_CAST(impl_, CModule_cls_impl)->get_result();
    }

}