# Faster-Rcnn-for-EOS-detection
## Use Faster RCNN to detect EOS cells.

## 文件内容：
- 1_point_label_creator.py：由于我们拥有两种不同标注方式，所以我们将黑色方框的标注方式转换成用绿点表示，统一成一种标注方式。
- 2_split_train_and_test.py: 该文件是将每张原始img和label文件切割成大小只有原图片面积1/4大小，同时将切割出来的文件的85%用于训练，15%用于检验结果
- 3_data_augmentation.py：该文件对于训练数据集进行大量数据增强，由于我们所检验的EOS细胞在最原始图片中尺寸是比较小的，在此我们将每张图片长宽resize成原来的4倍，同时切割成小图。然后由于我们所有图片中包含EOS细胞还是比较少，同时许多图片中EOS细胞的个数也比较少，所以在此我们根据对小目标检测数据增强的方式：分别对拥有许多EOS细胞的图片进行上采样，然后随机取一些没包含EOS细胞的图片中随机粘贴一些EOS细胞到上面的方式进行数据增强。最后我们是用keras的ImageDataGenerator来对图片进行旋转，裁剪，平移缩放等增强。
- 4_get_xml.py：我们将所得到的经过增强的imgs和labels进行位置框的检测同时还生成faster rcnn训练所需要的VOC数据集格式
- 5_faster_rcnn: 运用faster rcnn对数据进行训练和预测，具体的步骤和框架见[链接](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
- test_results.py: 用来预测结果以及对模型所结果进行评估

## 运行文件时生成的文件夹的内容：
1. source文件夹中包含医院所给的最原始的数据集，其中的imgs文件夹包含的是141张可能包含或不包含多个EOS细胞的图片，labels中每张图片名字与imgs图片对应，是其对应的用绿点或者黑色方框标注出来的label。
2. train文件夹包含85%用于训练的数据集，test文件夹包含15%用于测试的数据集。
3. aug文件夹是train文件夹中的图片经过data augmentation之后的结果，是用于最后文件的训练的
4. temp中文件是暂存的，我们不需要用到
5. results中masks文件包含我们用于制作训练标签的可视化之后的方框
6. datasets 中包含用于faster rcnn训练的标准数据格式。
