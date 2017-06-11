# testtheano
更新apt-get  
apt-get update  
  

GCC  
apt-get install build-essential  


# python2.7
1、下载官方源码:   
https://www.python.org/ftp/python/2.7.13/Python-2.7.13.tgz  
或https://www.python.org/ftp/python/2.7.13/Python-2.7.13.tar.xz  
2、解压后进入目录Python-2.7.13；  
3、配置并编译  
./configure —prefix=/usr/local/  CFLAGS=-fPIC  
make   
make install  
4、测试安装成功  
输入python，能进入python环境则成功  

# 安装pip  
wget https://bootstrap.pypa.io/get-pip.py   
python get-pip.py  
异常：zipimport.ZipImportError: can't decompress data; zlib not available  
解决方式:   
依赖包  apt-get install zlib*  
进入 Python安装包,修改Module路径的setup文件  
vimmodule/setup  
#zlibzlibmodule.c-I$(prefix)/include-L$(exec_prefix)/lib-lz  
去掉注释   
zlib zlibmodule.c-I$(prefix)/include-L$(exec_prefix)/lib-lz  
make && makeinstall  

异常： Could not find a version that satisfies the requirement pip (from versions: )  
No matching distribution found for pip  
受够了，直接apt-get install python-pip安装了  
然后自己给自己升级 pip  install -U pip  

#正则匹配删除文件  
ls |grep [A-Z] |xargs rm -rf  
  
### 修复yum  
export PATH=$PATH:/usr/local/lib/python2.7/dist-packages  
pip easy_install 软连接。  
# 安装依赖库：  
pip -V  
pip install numpy  
pip install scipy  
pip install nose  
pip install theano==0.8.2  
pip install Lasagne==0.1  
pip install pydot==1.1.0  
pip install simply  
pip install Pillow  

# 后台运行  
nohup Python manage.py runserver 0.0.0.0:9000 &  
  
# 生成训练集
进入training_data_gen 目录  
运行python image_dir max_size length -ignore_case=True -captcha_pattern='^\d+_(.*)\..+$'  
可以根据train_parse_args文件查看详细含义：
  image_dir为图片存储目录，
  max_size为多少张图片一个训练集文件，
  length为一张图片的字符数量，
  ignore_case为是否忽略字符大小写默认True，
  captcha_pattern为图片的名字的正则，group1为正确标签
