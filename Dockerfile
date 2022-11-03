# 拉取基础镜像，本镜像由华为云官方提供，公开可信
FROM swr.cn-north-4.myhuaweicloud.com/modelarts-job-dev-image/mindspore-ascend910-cp37-euleros2.8-aarch64-training:1.3.0-3.3.0-roma
MAINTAINER ***
WORKDIR /home/work

# 安装自定义版本Ascend Tookkit，这里使用的是5.0.4版本
USER root
COPY Ascend-cann-toolkit_5.0.4_linux-aarch64.run /home/work/Ascend-cann-toolkit_5.0.4_linux-aarch64.run
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


# RUN apt install -y libgmp-dev
# RUN sudo apt-get install m4
COPY gmp-6.1.2.tar.xz /home/work/gmp-6.1.2.tar.xz
RUN cd /home/work/
RUN tar -jvxf gmp-6.1.2.tar.xz
RUN ./gmp-6.1.2/configure --enable-cxx --prefix=/path_to_install --build=x86_64-linux
RUN make
RUN make check
RUN sudo make install

# 安装Mindspore
RUN pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-0.4.0-py3-none-any.whl
RUN pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-0.4.0-py3-none-any.whl
RUN pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-0.1.0-py3-none-any.whl
RUN pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.3.0/MindSpore/ascend/aarch64/mindspore_ascend-1.3.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -c "import mindspore;mindspore.run_check()"

USER work
# 安装Flask
RUN pip install Flask

# 安装依赖库
RUN pip install sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install loguru -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip uninstall urllib3 -y
RUN pip uninstall requests -y

# 拷贝应用服务代码进镜像里面
COPY --chown=work:work model /home/model

# 制定启动命令
CMD python3 /home/model/service.py