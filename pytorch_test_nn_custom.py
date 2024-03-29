import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(MyModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)


if __name__ == "__main__":
    net = MyModule(num_inputs=2, num_classes=5)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(net)
    print(out)
