import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):  # 输入值，输入的大小，输出的大小，激励函数
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 定义矩阵 随机变量生成初始的时候会比全0的好
    # 定义weights为一个in_size行, out_size列的随机变量矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 推荐初始值不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 当激励函数为None时，输出就是当前的预测值——Wx_plus_b
    # 不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。
    if activation_function is None:  # 线性关系，就不需要再加非线性方程
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 加一个维度  转换成300行1列
# print(x_data)
noise = np.random.normal(0, 0.05, x_data.shape)  # 期望，方差，格式
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # 隐藏层
prediction = add_layer(l1, 10, 1, activation_function=None)  # 输出层 输入10层，输出1层

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))  #

train_step = tf.train.GradientDescentOptimizer(0.08).minimize(loss)
# 优化器    目标，最小化误差

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)

for i in range(1500):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])  # 去除掉lines的第一个单位
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # prediction.append(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        # 每隔50次训练刷新一次图形，用红色、宽度为5的线来显示我们的预测数据和输入之间的关系，并暂停0.3s。
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.3)

plt.show()