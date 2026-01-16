import torch

# x = torch.tensor([1, 2, 3, 4, 5])
# y = torch.tensor([10, 20, 30, 40, 50])

# condition = x > 3
# # 替换不符合要求的元素
# result = torch.where(condition, x, y)

# print(result)
# # （10, 20, 30,  4,  5）


# t = torch.arange(0, 12, 2)
# print(t)  # tensor([ 0,  2,  4,  6,  8, 10])

# t = torch.arange(5, 0, -1)
# print(t)  # tensor([5, 4, 3, 2, 1])

# v1 = torch.tensor([1, 2, 3])
# v2 = torch.tensor([4, 5, 6])
# result = torch.outer(v1, v2)
# print(result)
# # tensor([[ 4,  5,  6],
# #         [ 8, 10, 12],
# #         [12, 15, 18]])

# t1=torch.tensor([[1,2,3],[4,5,6]])
# t2=torch.tensor([[7,8,9],[10,11,12]])
# result=torch.cat((t1,t2),dim=0)
# print(result)  

# result2=torch.cat((t1,t2),dim=1)
# print(result2)  

t1 = torch.tensor([1, 2, 3])
t2 = t1.unsqueeze(0)

print(t2)
print(t2.shape)