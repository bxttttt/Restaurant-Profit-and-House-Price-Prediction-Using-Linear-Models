import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

filename = "ex1data2.txt"
x = []
xx = []
y = []
with open(filename, 'r') as lines:
    for line in lines:
        line = line.split(',')
        # print(line)
        i = 0
        for data in line:
            if i == 0:
                x1 = int(data)
                xx.append(x1)
                i += 1
            elif i == 1:
                x2 = int(data)
                xx.append(x2)
                i += 1
            else:
                x.append(xx)
                xx = []
                y1 = int(data)
                y.append(y1)

# print(x)
# print(y)
num_training = int(len(x) * 0.8)
training_x = []
training_y = []
test_x = []
test_y = []
for i in range(0, len(x)):
    if i < num_training:
        training_x.append(x[i])
        training_y.append(y[i])
    else:
        test_x.append(x[i])
        test_y.append(y[i])

training_x = np.array(training_x)
training_y = np.array(training_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(training_x, training_y)
training_y_pred = linear_regressor.predict(training_x)
print("k:")
print(linear_regressor.coef_)
print("b:")
print(linear_regressor.intercept_)

# print(training_x)
x1 = []
x2 = []
for data in training_x:
    x1.append(data[0])
    x2.append(data[1])

plt.figure()
plt.scatter(x1, training_y, color="red")
plt.plot(x1, training_y_pred, color="black")
plt.title("training data(面积-价格)")
plt.show()

plt.figure()
plt.scatter(x2, training_y, color="red")
plt.plot(x2, training_y_pred, color="black")
plt.title("training data(卧室数-价格)")
plt.show()

test_y_pred = linear_regressor.predict(test_x)

xt1 = []
xt2 = []
for data in test_x:
    xt1.append(data[0])
    xt2.append(data[1])

plt.figure()
plt.scatter(xt1, test_y, color="red")
plt.plot(xt1, test_y_pred, color="black")
plt.title("test data(面积-价格)")
plt.show()

plt.figure()
plt.scatter(xt2, test_y, color="red")
plt.plot(xt2, test_y_pred, color="black")
plt.title("test data(卧室数-价格)")
plt.show()

final_test=[[2000,1]]
final_test=np.array(final_test)
print("预测结果：")
print(linear_regressor.predict(final_test))