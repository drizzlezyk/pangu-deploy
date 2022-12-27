## Pangu Deploy

#### 文件结构说明
- pcl_pangu文件夹 基于SDK pcl_pangu修改
- service_deploy.py和service_multi.py 主要定义基于Flask的服务
- Dockerfile 部署环境的镜像构建文件

#### 部署
基于SDK pcl_pangu, https://openi.pcl.ac.cn/PCL-Platform.Intelligence/pcl_pangu
在部署中遇到若干问题，对pcl_pangu部分源码进行修改：
1. setup_args()里的覆盖进行修改，除了input，其他参数采用命令行参数传递
2. load model和predict的解耦
3. tokenier的路径，设置为./xxx/vocab
4. 去掉online模块和 modle里pytorch模块，修改相应的init
5. 在MindSpore1.3将盘古的512个模型文件合并为一个
6. 由于镜像环境原因，修改为使用MindSpore1.7进行推理
