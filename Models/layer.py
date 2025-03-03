

from cv2 import mean
from sympy import print_rcode
import torch
import torch.nn as nn

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, input, gama, thre):
    def forward(ctx, input, gama):
        #out = ((input - thre) >= 0).float() - ((-1 * thre - input) >= 0).float()
        out = ((input) >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out
    '''def forward(ctx, input, gama):
            if input >= 0:
                out = (input >= 0).float()
            else
                out = -(input < 0).float()
            
            L = torch.tensor([gama])
            ctx.save_for_backward(input, out, L)
            return out
                '''
    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class IF(nn.Module):
    def __init__(self, T=0, Tp=0, L=8, thresh=8.0, tau=1., gama=1.0):
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.Tp = Tp
        self.loss = 0
        self.threshval = thresh

    def forward(self, x):

        if self.T > 0:
            thre = self.thresh.data
            tp = self.Tp
            x = self.expand(x)
            mem = 0.5 * thre
            
            # Cray  初始化membrane_lower 全为1
            membrane_lower = torch.ones_like(x[0,...])

            # 初始化标志位，用于判断是否产生负脉冲。
            # 注意要把T的向量位减掉，否则和下面spike的向量形式不一样
            flag = torch.zeros_like(x[0,...])
            
            spike_pot = []
            #for t in range(self.T):
            for t in range(self.T + self.Tp):
                # mem = mem + x[t, ...]
                if t>=tp:           #Cray 修改为下面4行。预处理后，正式处理时，重新从x[0,...]开始
                    mem = mem + x[t-tp, ...]  
                else:
                    mem = mem + x[t, ...]

                posspike = self.act(mem - thre, self.gama) * thre
                # 添加负脉冲计算，
                # 直接取了thre的负值，是否应该和正脉冲的thre值一样，在act中进行backword？
                negspike=mem.le(thre*(-1)).float() * thre*(-1)
    
                # 根据标志位确定是否产生负脉冲
                compare = torch.where(flag > 0, torch.ones_like(flag), torch.zeros_like(flag))
                negspike = negspike * compare
                # 综合正负脉冲得到最终脉冲向量
                spike = posspike + negspike
                
                mem = mem - spike
    
                # 更新标志位
                flag= flag + spike

                #CRay 这句移到if t >= tp:判断里面，计算完membrane_lower后，用其更新spike在附加到spike_pot中
                #spike_pot.append(spike)   

                #if t == tp:
                if t == (tp-1):   #Cray 修改为tp-1
                    membrane_lower = torch.where(mem > 1e-3, torch.ones_like(mem), torch.zeros_like(mem))
                    mem = 0.5 * thre
                    #spike_pot = [] #Cray 增加一句，预处理结束后，重新初始化spike_pot
                #if t > tp:
                if t >= tp:       #Cray 修改为t > = tp
                    #mem = mem * membrane_lower
                    spike = spike * membrane_lower   #Cray 输出脉冲乘以membrane_lower
                    spike_pot.append(spike)          #Cray 从上面位置移到此处
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)

        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh
        return x


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x
