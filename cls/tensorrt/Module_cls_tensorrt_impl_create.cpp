#include "tensorrt/Module_cls_tensorrt_impl.h"
#include "Module_cls.h"

namespace tensorrt_cls
{
    CModule_cls::CModule_cls()
    {
        impl_ = new ALG_ENGINE_IMPL(cls, tensorrt);
    }
}