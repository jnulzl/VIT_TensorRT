# TensorRT Classification

## 开发环境

- Ubuntu 18.04 + 

- CUDA 11.x

- TensorRT 8.x

- cudnn 8.x

- OpenCV_CUDA 4.x

## build

- 编译demo

```shell
# build demo
>>mkdir build && cd build
>>cmake —DCMAKE_BUILD_TYPE=Release -DCUDA_HOME="YOUR_CUDA_HOME" -DCUDNN_HOME="YOUR_CUDNN_HOME" -DTRT_HOME="YOUR_TENSORRT_HOME" -DOpenCV_DIR="YOUR_OPENCV_DIR" ..
>>make VERBOSE=1 -j8 
```

编译好的文件位于`$YOUR_ROOT/bin/Linux`

- 运行demo
- 
```shell
>>cd $YOUR_ROOT/bin/Linux
>>./demo ....
```

## 其它说明

- **如果你用的CUDA, OpenCV, TensorRT或者cudnn与上述描述的不一致可能存在细微的api差异，可在此基础上稍微修改即可**。