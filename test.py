# %%
import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

# %%
#创新之处在于更庞大，更深层的网络的测试
#某种程度上定义了深层网络的超参数设计范式，池化层采用与核size类似的stride，从浅到深核的尺寸逐渐下降，输出的channel逐渐加深，前两个维度则逐渐减小，最后会接入一个MLP用于分类输出。

batch_size=128
lr=0.01
num_epochs=10

class Log_s(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        return torch.max(torch.zeros_like(X),torch.log10(X+1))

net=nn.Sequential(nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),Log_s(),   #size=54
                  nn.MaxPool2d(kernel_size=3,stride=2),     #size=26
                  nn.Conv2d(96,256,kernel_size=5,padding=2),Log_s(),    #26
                  nn.MaxPool2d(kernel_size=3,stride=2),     #12
                  nn.Conv2d(256,384,kernel_size=3,padding=1),  #12
                  nn.Conv2d(384,384,kernel_size=3,padding=1),  #12
                  nn.Conv2d(384,256,kernel_size=3,padding=1),  #12
                  nn.MaxPool2d(kernel_size=3,stride=2),    #5
                  nn.Flatten(),
                  nn.Linear(5*5*256,4096),Log_s(),nn.Dropout(0.5),
                  nn.Linear(4096,4096),Log_s(),nn.Dropout(0.5),
                  nn.Linear(4096,10))


def init_weight(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weight)

loss=nn.CrossEntropyLoss()

updater=torch.optim.SGD(net.parameters(),lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)


# %%
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# %%
def train_ch6(net, train_iter, test_iter, optimizer,loss,num_epochs, device):
    print('training on', device)
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

# %%
train_ch6(net,train_iter,test_iter,updater,loss,num_epochs,torch.device('cuda'))

# %%



