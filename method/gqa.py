import torch
import torch.nn as nn

# dropout_layer = nn.Dropout(p=0.5)

# t1=torch.Tensor([1,2,3])
# t2=dropout_layer(t1)
# # 这里Dropout丢弃了1，为了保持期望不变，将1和3扩大两倍
# print(t2)

# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1, 2, 3])  # shape: (3,)

# t2 = torch.Tensor([[1, 2, 3]])  # shape: (1, 3)
# # 这里应用的w和b是随机的，真实训练里会在optimizer上更新
# output2 = layer(t1)             # shape: (1, 5)
# print(output2)

t1=torch.Tensor([[1,2,3],[4,5,6]])
t1=t1.transpose(0,1)
print(t1)