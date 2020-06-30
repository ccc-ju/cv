# !/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@author:ccc-ju
@software:PyCharm
@time:2020/6/23 11:19
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_data():
	# 本函数生成0-9，10个数字的图片矩阵
	image_data = []
	num_0 = torch.tensor(
		[[0, 0, 1, 1, 0, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 0, 1, 1, 0, 0],
		 [0, 0, 0, 0, 0, 0]])
	image_data.append(num_0)
	num_1 = torch.tensor(
		[[0, 0, 0, 1, 0, 0],
		 [0, 0, 1, 1, 0, 0],
		 [0, 0, 0, 1, 0, 0],
		 [0, 0, 0, 1, 0, 0],
		 [0, 0, 1, 1, 1, 0],
		 [0, 0, 0, 0, 0, 0]])
	image_data.append(num_1)
	num_2 = torch.tensor(
		[[0, 0, 1, 1, 0, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 0, 0, 1, 0, 0],
		 [0, 0, 1, 0, 0, 0],
		 [0, 1, 1, 1, 1, 0],
		 [0, 0, 0, 0, 0, 0]])
	image_data.append(num_2)
	num_3 = torch.tensor(
		[[0, 0, 1, 1, 0, 0],
		 [0, 0, 0, 0, 1, 0],
		 [0, 0, 1, 1, 0, 0],
		 [0, 0, 0, 0, 1, 0],
		 [0, 0, 1, 1, 0, 0],
		 [0, 0, 0, 0, 0, 0]])
	image_data.append(num_3)
	num_4 = torch.tensor(
		[
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 1, 1, 0],
			[0, 0, 1, 0, 1, 0],
			[0, 1, 1, 1, 1, 1],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0]])
	image_data.append(num_4)
	num_5 = torch.tensor(
		[
			[0, 1, 1, 1, 0, 0],
			[0, 1, 0, 0, 0, 0],
			[0, 1, 1, 1, 0, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 1, 1, 1, 0, 0],
			[0, 0, 0, 0, 0, 0]])
	image_data.append(num_5)
	num_6 = torch.tensor(
		[[0, 0, 1, 1, 0, 0],
		 [0, 1, 0, 0, 0, 0],
		 [0, 1, 1, 1, 0, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 0, 1, 1, 0, 0],
		 [0, 0, 0, 0, 0, 0]])
	image_data.append(num_6)
	num_7 = torch.tensor(
		[
			[0, 1, 1, 1, 1, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 0]])
	image_data.append(num_7)
	num_8 = torch.tensor(
		[[0, 0, 1, 1, 0, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 0, 1, 1, 0, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 0, 1, 1, 0, 0],
		 [0, 0, 0, 0, 0, 0]])
	image_data.append(num_8)
	num_9 = torch.tensor(
		[[0, 0, 1, 1, 1, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 0, 1, 1, 1, 0],
		 [0, 1, 0, 0, 1, 0],
		 [0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0]])
	image_data.append(num_9)
	return image_data


def get_feature(x):
	row_feature = [0, 0, 0, 0, 0, 0, 0]
	col_feature = [0, 0, 0, 0, 0, 0, 0]
	# 下面添加提取图像x的特征feature的代码
	# row, col = x.shape

	# for i in range(row):
	# 	sum = 0
	# 	for j in range(col):
	# 		sum += x[i][j]
	# 	feature_1[i] = sum

	row_feature = torch.sum(x, 1)
	col_feature = torch.sum(x, 0)
	feature = torch.cat((row_feature, col_feature))
	return feature


def model(feature):
	y = -1
	# 下面添加对feature进行决策的代码，判定出feature 属于[0,1,2,3,...9]哪个类别
	num_0 = [2, 2, 2, 2, 2, 0, 0, 3, 2, 2, 3, 0]
	num_1 = [1, 2, 1, 1, 3, 0, 0, 0, 2, 5, 1, 0]
	num_2 = [2, 2, 1, 1, 4, 0, 0, 2, 3, 3, 2, 0]
	num_3 = [2, 1, 2, 1, 2, 0, 0, 0, 3, 3, 2, 0]
	num_4 = [1, 2, 2, 5, 1, 0, 0, 1, 2, 2, 5, 1]
	num_5 = [3, 1, 3, 1, 3, 0, 0, 4, 3, 3, 1, 0]
	num_6 = [2, 1, 3, 2, 2, 0, 0, 3, 3, 3, 1, 0]
	num_7 = [4, 1, 1, 1, 1, 0, 0, 1, 1, 4, 2, 0]
	num_8 = [2, 2, 2, 2, 2, 0, 0, 2, 3, 3, 2, 0]
	num_9 = [3, 2, 3, 2, 1, 0, 0, 2, 2, 2, 5, 0]
	if num_0 == feature:
		y = 0
	if num_1 == feature:
		y = 1
	if num_2 == feature:
		y = 2
	if num_3 == feature:
		y = 3
	if num_4 == feature:
		y = 4
	if num_5 == feature:
		y = 5
	if num_6 == feature:
		y = 6
	if num_7 == feature:
		y = 7
	if num_8 == feature:
		y = 8
	if num_9 == feature:
		y = 9
	return y


if __name__ == "__main__":

	image_data = generate_data()
	# 打印出0的图像
	print("数字0对应的图片是:")
	print(image_data[0])
	print("-" * 20)

	# 打印出8的图像
	print("数字8对应的图片是:")
	print(image_data[8])
	print("-" * 20)

	# 对每张图片进行识别
	print("对每张图片进行识别")
	for i in range(0, 10):
		x = image_data[i]
		# 对当前图片提取特征
		feature = get_feature(x)
		# 对提取到得特征进行分类
		y = model(feature)
		# 打印出分类结果
		print("图像[%s]得分类结果是:[%s],它得特征是[%s]" % (i, y, feature))