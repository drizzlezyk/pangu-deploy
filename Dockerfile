# 拉取基础镜像，本镜像由华为云官方提供，公开可信
FROM swr.cn-north-4.myhuaweicloud.com/modelarts-job-dev-image/mindspore-ascend910-cp37-euleros2.8-aarch64-training:1.3.0-3.3.0-roma
MAINTAINER ***
WORKDIR /home/work

# 安装自定义版本Ascend Tookkit，这里使用的是5.0.4版本
USER root
COPY --chmod=a+x Ascend-cann-toolkit_5.0.4_linux-aarch64.run /home/work/Ascend-cann-toolkit_5.0.4_linux-aarch64.run
RUN /usr/local/Ascend/nnae/latest/script/uninstall.sh
RUN /home/work/Ascend-cann-toolkit_5.0.4_linux-aarch64.run --full
RUN rm -rf /home/work/Ascend-cann-toolkit_5.0.4_linux-aarch64.run

# 设置环境变量
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH \
    PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH \
    PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:$PATH \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit \
    ASCEND_AUTOML_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools

# 安装Mindspore
USER work
RUN pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-0.4.0-py3-none-any.whl \
                /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-0.4.0-py3-none-any.whl \
                /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-0.1.0-py3-none-any.whl \
                https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindSpore/ascend/aarch64/mindspore_ascend-1.6.1-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装Flask
RUN pip install Flask

# 安装依赖库
RUN pip install Pillow
RUN pip install loguru
RUN pip install typing
RUN pip install argparse
RUN pip install sentencepiece
RUN pip install jieba

# 拷贝应用服务代码进镜像里面
COPY model /home/model

# 制定启动命令
CMD python3 /home/model/service.py