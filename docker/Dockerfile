FROM registry.cn-beijing.aliyuncs.com/shengteng_remote/tianchi:0.0

# 将程序复制容器内，表示在/workspace 路径下
COPY . /workspace

# 确定容器启动时程序运行路径
WORKDIR /workspace

RUN cd code/mmseg_codes \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e . && python setup.py develop \
 && cd ../..

ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8

CMD ["bash", "run.sh"]

