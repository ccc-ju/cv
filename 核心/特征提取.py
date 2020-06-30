import torch
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

image_data = generate_data()
for i in range(0, 10):
	x = image_data[i]
	# 对当前图片提取特征
	feature = get_feature(x)
	print(i, '\t',feature)