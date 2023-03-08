import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from sklearn import linear_model

A = np.array([[2,9,7,9,11,16,25,23,22,27,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

#Đưa ra số hàng của ma trận A
m = A.shape[0]

#
def cost(x):
    return 0.5/m*(np.linalg.norm(A.dot(x)-b,2)**2)

def gradient(x):
    return 1/m*A.T.dot(A.dot(x)-b)

def check_grap(x):
    eps = 1e-4
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        g[i] = (cost(x+eps)-cost(x-eps))/(2*eps)

    g_grad = gradient(x)
    if np.linalg.norm(g-g_grad) >  1e-7:
        print("WARNING: Check gradient fuction!")
def gradient_descent(x_init, learning_rate, iteration):
    x_list = [x_init]
    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*gradient(x_list[-1])

        x_list.append(x_new)
    return x_list

fig1 = plt.figure("Ngọc Sơn-Linear Regression")
ax = plt.axes(xlim=(-10,60), ylim=(-1,20))
plt.plot(A,b, 'ro')

lr = linear_model.LinearRegression()
lr.fit(A,b)
x0_gd = np.linspace(1,46,2)
y0_skl = lr.coef_[0][0]*x0_gd + lr.intercept_[0]
plt.plot(x0_gd, y0_skl, color = "green")

# Thêm vector 1 vào A
ones = np.ones((A.shape[0],1), dtype= np.int8)
A = np.concatenate((ones, A), axis = 1)

# Random initial line
x_init = np.array([[1.0],[2.0]])
y0_init = x_init[1]*x0_gd + x_init[0]
plt.plot(x0_gd,y0_init, color="purple")
check_grap(x_init)

# Thuật toán
iteration = 90
learning_rate = 0.0001
x_list = gradient_descent(x_init, learning_rate, iteration)
# for i in range(len(x_list)):
# 	y0_x_list = x_list[i][1]*x0_gd + x_list[i][0]
# 	plt.plot(x0_gd, y0_x_list, color = "purple")

# Tạo animation
line , = ax.plot([],[], color = "blue")
def update(i):
	y0_gd = x_list[i][1][0]*x0_gd + x_list[i][0][0]
	line.set_data(x0_gd, y0_gd)
	return line,

iters = np.arange(1, len(x_list), 1)
line_ani = ani.FuncAnimation(fig1, update, iters, interval = "50", blit = True)

# Chú thích
plt.legend(("Value in each GD iteration", "Solution by formular", "Inital value for GD"), loc=(0.52, 0.01))
ltext = plt.gca().get_legend().get_texts()

print(x_list)
plt.show()
