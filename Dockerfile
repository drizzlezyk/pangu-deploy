FROM swr.cn-north-4.myhuaweicloud.com/modelarts-job-dev-image/mindspore-ascend910-cp37-euleros2.8-aarch64-training:1.3.0-3.3.0-roma

# 安装自定义版本Ascend Tookkit，这里使用的是5.0.4版本
USER root
COPY Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run /home/work/Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run
RUN /usr/local/Ascend/nnae/latest/script/uninstall.sh
RUN chmod a+x /home/work/Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run
RUN /home/work/Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run --full
RUN rm -rf Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run

# 设置环境变量
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH \
    PYTHONPAiTH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH \
    PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:$PATH \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit \
    ASCEND_AUTOML_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools

# RUN apt install -y libgmp-dev

# 安装Mindspore
USER work
RUN pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-0.4.0-py3-none-any.whl \
                /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-0.4.0-py3-none-any.whl \
                /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-0.1.0-py3-none-any.whl \
                https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.7.0/MindSpore/ascend/aarch64/mindspore_ascend-1.7.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple


# 安装Flask
RUN pip install Flask -i https://pypi.tuna.tsinghua.edu.cn/simple

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

RUN pip install multiprocess \
		gevent \
		gunicorn -i https://pypi.tuna.tsinghua.edu.cn/simple


# 拷贝应用服务代码进镜像里面
COPY --chown=work:work model /home/model

# 制定启动命令
# CMD python3 /home/model/service_deploys.py --tokenizer_path /home/model/tokenizer/vocab  --load_ckpt_local_path /home/model/one_ckpt/ --distribute false --use_past false

CMD gunicorn -w 1 -k gevent -b 0.0.0.0:8080 -t 1200 service_multi:APP
