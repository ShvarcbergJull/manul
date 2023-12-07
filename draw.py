import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# filename = "swiss_res.txt"
# filename = "experiment1_result9.txt"
# filename = "huhuhf_2.txt"
filename = "result_exp1_manul2.txt"

with open(filename, "r") as fl:
    tet = fl.read()

data = tet.split("\n")
# print(data[1])
data = [ast.literal_eval(data[0]), ast.literal_eval(data[1])]
# data = [ast.literal_eval(data[0]), ast.literal_eval(data[1]), ast.literal_eval(data[2]), ast.literal_eval(data[3])]
av1 = [np.average(elem) for elem in data[0]]
av2 = [np.average(elem) for elem in data[1]]
# av3 = [np.average(elem) for elem in data[2]]
arr1 = np.array(data[0])
arr2 = np.array(data[1])
# arr3 = np.array(data[2])
# arr4 = np.array(data[3])

print("DRAW AVERAGe")
plt.plot(av1, label="without")
plt.plot(av2, label="with")
# plt.plot(av3, label="jst")
plt.legend()
plt.show()

print("DRAW EACH CLASS")
plt.plot(arr1, label="without")
plt.plot(arr2, label="with")
plt.legend()
plt.show()

print("BOXPLOTS")
# box_data = np.concatenate((arr1, arr2, arr3), axis=1)
# plt.boxplot(box_data, labels=["1 класс,\nбез графа", "2 класс,\nбез графа", "1 класс,\nс графом", "2 класс,\nс графом", "1 класс", "2 класс"])

box_data = np.concatenate((arr1, arr2), axis=1)
# plt.boxplot(box_data, labels=["1 класс,\nbase", "2 класс,\nbase", "1 класс,\nфрейм", "2 класс,\nфрейм", "1 класс,\nманул", "2 класс,\nманул", "1 класс,\nбез графа", "2 класс,\nбез графа"])
plt.boxplot(box_data, labels=["1 класс,\nbase", "2 класс,\nbase", "1 класс,\nman", "2 класс,\nman"])
# plt.boxplot(arr2)
plt.show()

# with open("experiment1_result_mi.txt", "r") as fl:
#     tet = fl.read()

# data = tet.split("\n")
# data = [ast.literal_eval(data[0]), ast.literal_eval(data[1])]
# av3 = [np.average(elem) for elem in data[0]]
# av4 = [np.average(elem) for elem in data[1]]
# arr3 = np.array(data[0])
# arr4 = np.array(data[1])

# print("BOXPLOTS2")
# box_data = np.concatenate((arr2, arr4), axis=1)
# plt.boxplot(box_data)
# # plt.boxplot(arr2)
# plt.show()

# print("DRAW AVERAGe")
# plt.plot(av4, label="frame")
# plt.plot(av2, label="my")
# plt.legend()
# plt.show()