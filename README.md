# RNN-Sequential-Mnist
Tensorflow eager implementation to solving Sequential-Mnist classification problem using Recurrent Neural network

## 具体内容参照
`https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-2-f7e5ece849f5`
但该博客中的实现方式为原始的Tensorflow操作。这里对其进行了简化，使用 `TensorFlow eager execution`

## 原理
1. 用RNN来分类Mnist数据时，对于训练数据`(batch_size, height, weight)` 可以将其看做自然语言处理中的`（batch_size, sequence_length, embedding_dim）`，
相当于在RNN的每个时间步输入一张图片一行中的28个像素，分为28个时间步依次进行输入。在最后一个时间步计算损失。
2. 也可以考虑在每个时间步输入一个像素，但这样序列长度太长，不易收敛。

## 原理示意图
![avatar](https://cdn-images-1.medium.com/max/2000/1*a5iGm8sByBwvUzH0kxcu3Q.jpeg)

## 训练结果
在前10个周期的训练中，测试集可以达到90%的准确率
```
Epoch : 0 , train loss : 1.9766251
**Epoch : 0 , valid acc: 57.82000000000001 %
Epoch : 1 , train loss : 1.2181218
Epoch : 2 , train loss : 0.92483646
**Epoch : 2 , valid acc: 77.84 %
Epoch : 3 , train loss : 0.769496
Epoch : 4 , train loss : 0.65395266
**Epoch : 4 , valid acc: 85.34 %
Epoch : 5 , train loss : 0.5726876
Epoch : 6 , train loss : 0.50582236
**Epoch : 6 , valid acc: 88.47 %
Epoch : 7 , train loss : 0.45860845
Epoch : 8 , train loss : 0.41900662
**Epoch : 8 , valid acc: 90.49000000000001 %
Epoch : 9 , train loss : 0.39168894
```





