import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    if p is not None:
        return p
    if isinstance(k, tuple):
        k = k[0]
    return k // 2



class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BottleNeck(nn.Module):
    def __init__(self, c1, c2, e, shortcut = True):
        super().__init__()
        hidden = int(c2*e)
        self.cv1 = Conv(c1, hidden, k = 1, s=1)
        self.cv2 = Conv(hidden, c2, k = 3, p = 1)
        self.add = shortcut and c1 == c2


    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.add:
            return x + y
        else:
            return y
        
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)

class C2f(nn.Module):
    def __init__(self, c1, c2 , n = 1, shortcut = False , e = 1):
        super().__init__()
        self.c = int(c2*e)
        self.cv1 = Conv(c1,2*self.c, k=1,s=1)
        self.cv2 = Conv((n+2)*self.c, c2, k = 1, s =1)
        self.m = nn.ModuleList([BottleNeck(self.c, self.c ,e , shortcut) for _ in range(n)])


    def forward(self, x):
        y = list(self.cv1(x).chunk(2,1))
        y.extend([m(y[-1]) for m in self.m])
        return self.cv2(torch.cat(y, 1))

class Upsample(nn.Module):
    def __init__(self, scale_factor = 2, mode = 'nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode = mode)

    def forward(self, x):
        return self.upsample(x)



class SPPF(nn.Module):
    def __init__(self, c1, c2, k = 5):
        super().__init__()

        c_ = c1 // 2

        self.cv1 = Conv(c1, c_, k = 1 , s = 1)
        self.cv2 = Conv(c_ * 4, c2, k = 1, s = 1)
        self.MaxPooling = nn.MaxPool2d(kernel_size = k, stride = 1, padding= k//2)


    def forward(self, x):
        y = [self.cv1(x)]
        y.extend([self.MaxPooling(y[-1]) for _ in range(3)])
        return self.cv2(torch.cat(y,1))


class Detect(nn.Module):
    def __init__(self, num_classes = 2, reg_max = 16, ch = [256 , 512 , 1024]):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = [8, 16 , 32]

        self.cv2 = [(Conv(p, self.num_classes + 4*self.reg_max, 1)) for p in ch]



    def forward(self,x):
        outputs = []

        for i in range(3):
            out = self.cv2[i](x[i])
            outputs.append(out)


        return outputs


class YOLOv8(nn.Module):
    def __init__(self,confidence_threshold,iou_threshold, num_classes = 2, reg_max = 16):
        super().__init__()
        #Backbone
        self.cv0 = Conv(3,64,k = 3, s = 2, p = 1)
        self.cv1 = Conv(64,128,k = 3, s = 2, p = 1)
        self.cv2 = C2f(128,128,n = 3 ,shortcut=True,e=1)
        self.cv3 = Conv(128,256,k = 3, s = 2, p = 1)
        self.cv4 = C2f(256,256,n = 6 ,shortcut=True,e=1)
        self.cv5 = Conv(256,512,k = 3, s = 2, p = 1)
        self.cv6 = C2f(512,512,n = 6 ,shortcut=True,e=1)
        self.cv7 = Conv(512,1024,k = 3, s = 2, p = 1)
        self.cv8 = C2f(1024,1024,n = 6 ,shortcut=True,e=1)
        self.cv9 = SPPF(1024,1024, k = 5)


        #Neck
        self.cv10 = Upsample()
        self.cv11 = Concat()
        self.cv12 = C2f(512+1024,512,n = 3,shortcut=False,e=1)
        self.cv13 = Upsample()
        self.cv14 = Concat()
        self.cv15 = C2f(512+256,256,n = 3,shortcut=False,e=1)
        self.cv16 = Conv(256,256,k = 3 , s = 2, p = 1)
        self.cv17 = Concat()
        self.cv18 = C2f(512+256, 512,n = 3, shortcut=False , e= 1)
        self.cv19 = Conv(512, 512, k = 3, s = 2, p = 1)
        self.cv20 = Concat()
        self.cv21 = C2f(1024+512,1024, n = 3, shortcut= False, e= 1)

        #Head
        self.cv22 = Detect(num_classes=num_classes, reg_max=reg_max, ch=[256, 512, 1024])
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def forward(self, x):
        x1 = self.cv0(x)
        x2 = self.cv1(x1)
        x3 = self.cv2(x2)
        x4 = self.cv3(x3)
        x5 = self.cv4(x4)
        x6 = self.cv5(x5)
        x7 = self.cv6(x6)
        x8 = self.cv7(x7)
        x9 = self.cv8(x8)
        x10 = self.cv9(x9)

        # Neck
        u1 = self.cv10(x10)
        c1 = self.cv11([u1, x7])
        n1 = self.cv12(c1)

        u2 = self.cv13(n1)
        c2 = self.cv14([u2, x5])
        n2 = self.cv15(c2)

        d1 = self.cv16(n2)
        c3 = self.cv17([d1, n1])
        n3 = self.cv18(c3)

        d2 = self.cv19(n3)
        c4 = self.cv20([d2, x10])
        n4 = self.cv21(c4)

        # Head
        outputs = self.cv22([n2, n3, n4])
        return outputs
