#include "Module_vit_impl.h"
#include "Module_vit.h"
#include "ai_alg_version.h"

#include "alg_define.h"
#include "debug.h"

namespace tensorrt_vit
{
    CModule_vit::~CModule_vit()
    {
        if(ANY_POINTER_CAST(impl_, CModule_vit_impl))
        {
            delete ANY_POINTER_CAST(impl_, CModule_vit_impl);
        }
    }

    void CModule_vit::init(const BaseConfig &config)
    {
        print_version();
        ANY_POINTER_CAST(impl_, CModule_vit_impl)->init(config);
    }

    void CModule_vit::deinit()
    {
        ANY_POINTER_CAST(impl_, CModule_vit_impl)->deinit();
#if defined(ALG_DEBUG) || defined(AI_ALG_DEBUG)
        AIALG_PRINT("release success!\n");
#endif
    }

    void CModule_vit::process_batch(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        ANY_POINTER_CAST(impl_, CModule_vit_impl)->process_batch(imageInfos, batch_size);
    }

    const SegmentResult* CModule_vit::get_result()
    {
        return ANY_POINTER_CAST(impl_, CModule_vit_impl)->get_result();
    }

    const BaseConfig* CModule_vit::get_config() const
    {
        return ANY_POINTER_CAST(impl_, CModule_vit_impl)->get_config();
    }
}
