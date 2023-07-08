import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

filename1 = "ex1data1.txt"

x = []
y = []
with open(filename1, 'r') as lines:
    for line in lines:
        line = line.split(',')
        i = 0
        for data in line:
            if i == 0:
                xt = float(data)
                i += 1
                x.append(xt)
            else:
                yt = float(data)
                y.append(yt)
    # x为面积,y为年收入
    print(x)
    print(y)

num_training_x = int(len(x) * 1)
training_x = []
training_y = []

for i in range(0, len(x)):
    training_x.append(x[i])
    training_y.append(y[i])

training_x = np.array(training_x).reshape((len(training_x), 1))
training_y = np.array(training_y)

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(training_x, training_y)
training_y_pred = linear_regressor.predict(training_x)

print("k:")
print(linear_regressor.coef_)
print("b:")
print(linear_regressor.intercept_)

plt.figure()
plt.scatter(training_x, training_y, color='red')
plt.plot(training_x, training_y_pred, color='black')
plt.title("餐厅利润问题")
plt.show()

final_test = [3.1415]
final_test = np.array(final_test).reshape((len(final_test), 1))

print("作业1预测的结果：")
print(linear_regressor.predict(final_test))
print('MSE:', mean_squared_error(training_y, training_y_pred))
print('R2:', r2_score(training_y, training_y_pred))
