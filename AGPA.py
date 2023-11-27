class AGPA(nn.Module):

    def __init__(self, c1,c2, k_size=5):
        super(AGPA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.sigma = nn.Parameter(torch.randn(1), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)
        self.arfa = nn.Parameter(torch.randn(1), requires_grad=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        s = self.sig(self.sigma)
        b = self.sig(self.beta)
        f = self.sig(self.arfa)

        y = self.avg_pool(x)
        y1 = self.max_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)   # y = avg
        y1 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # y1 = max

        w = s*y+(1-s)*y1    # w = sigma*avg+(1-s)*max
        z = b*y-(1-b)*y1     # z = beta*avg-(1-b)*max

        out = self.sigmoid(f*w+(1-f)*z)

        return x * out.expand_as(x)