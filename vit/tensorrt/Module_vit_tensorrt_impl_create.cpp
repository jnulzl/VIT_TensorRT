#include "tensorrt/Module_vit_tensorrt_impl.h"
#include "Module_vit.h"

namespace tensorrt_vit
{
    CModule_vit::CModule_vit()
    {
        impl_ = new ALG_ENGINE_IMPL(vit, tensorrt);
    }
}