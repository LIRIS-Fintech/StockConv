

import torch
import torch.nn as nn
import torch.nn.functional as F

acv = nn.GELU()

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price).to(device)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio




class IntraStock(nn.Module):
    def __init__(self, in_channels, hid_channels, seq_len, kernel_size=1):
        super(IntraStock, self).__init__()

        self.depthwise1 = nn.Conv1d(in_channels, in_channels, kernel_size=3,
                                    dilation=1, groups=in_channels, padding="same")
        self.depthwise2 = nn.Conv1d(in_channels, in_channels, kernel_size=3,
                                    dilation=2, groups=in_channels, padding="same")
        self.pointwise = nn.Conv1d(in_channels, 1, kernel_size=1)  
        self.act = acv
        self.norm = nn.LayerNorm([in_channels, seq_len])
        
    def forward(self, x): # x: [num, time, feature]
        x = torch.transpose(x, -1, -2)       
        y1 = self.depthwise1(x)
        y2 = self.depthwise2(x)
        y1 = self.act(y1)
        y3 = self.depthwise2(y1)
        y1 = self.norm(y1)
        y2 = self.norm(y2)
        y3 = self.norm(y3)
        z1 = self.pointwise(y1)
        z2 = self.pointwise(y2)
        z3 = self.pointwise(y3)
        z = torch.cat([z1, z2, z3], dim=2)
        
        return z
        
        
class InterStock(nn.Module):
    def __init__(self, num=1026, hid_num=10, kernel_size=1, group=27):         
        super(InterStock, self).__init__()
        self.depthwise1 = nn.Conv1d(num, num, kernel_size=3, dilation=2, 
                                    groups=num, padding="same")
        self.depthwise2 = nn.Conv1d(num, num, kernel_size=3, dilation=4, 
                                    groups=num, padding="same")
        self.pointwise1 = nn.Conv1d(num, num, kernel_size=1, groups=79) 
        self.pointwise2 = nn.Conv1d(num, num, kernel_size=1, groups=158)    
        self.act = acv
        self.norm= nn.LayerNorm([num])
    
    def forward(self, x): 
        y = self.pointwise1(x)
        y = self.pointwise2(y)
        y1 = self.depthwise1(y)
        y2 = self.depthwise2(y1)
        z = torch.cat([y1, y2], dim=1)
        
        return z
        

class StockConvMixer(nn.Module):
    def __init__(self, in_channels, hid_channels, num, hid_num, seq_len, time_steps):
        super(StockConvMixer, self).__init__()
        self.intrastock = IntraStock(in_channels, hid_channels, seq_len)
        self.interstock = InterStock(num, hid_num)
        self.lin1 = nn.Linear(time_steps*3, 1)
        self.lin2 = nn.Linear(time_steps*6, 1)
        self.lin3 = nn.Linear(1,1)
        self.lin4 = nn.Linear(1,1)
    
    def forward(self, x):
        x = self.intrastock(x)
        x = torch.squeeze(x)
        y = self.interstock(x)
        y = self.lin2(y)
        z = self.lin1(x)
        out = self.lin3(y+z)
        
        return out
    
