#그래프 그리기 요호
import matplotlib.pyplot as plt
import numpy as np

array_len = 10000
var1 = np.zeros(array_len)
var2 = np.zeros(array_len)
var3 = np.zeros(array_len)
var4 = np.zeros(array_len)


def function_1(input):
    # Function 1
    for i in range(array_len):
        var1[i] = 3 + i * input
        var2[i] = 2 * var1[i]
    return var1, var2


def function_2(input):
    # Function 2
    for j in range(array_len):
        var3[j] = 3 + j * input
        var4[j] = 10 * var3[j]
    return var3, var4


Output = function_1(1)[0] + function_2(2)[0]
plt.plot(Output)
plt.show(Output.all())