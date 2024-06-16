import torch
from torch import nn
import torch.nn.functional as F

# lambda
# class LearnableWeight(nn.Module):
#     def __init__(self):
#         super(LearnableWeight, self).__init__()
#         # 设置可学习的参数，设置初始值为0.5
#         self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
#         self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

#     def forward(self, x1, x2):
#         out = x1 * self.w1 + x2 * self.w2
#         return out

# # 残差结构的系数
# class LearnableCoefficient(nn.Module):
#     def __init__(self):
#         super(LearnableCoefficient, self).__init__()
#         self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

#     def forward(self,x):
#         out = x * self.bias
#         return out

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_positional_encoding(self, T, C):
        position = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, C, 2) * -(torch.log(torch.tensor(10000.0)) / C))
        pos_enc = torch.zeros((T, C))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, input_tensor):
        B, T, C = input_tensor.size()
        position_enc = self.get_positional_encoding(T, C).repeat(B, 1, 1).to(input_tensor.device)
        output_tensor = input_tensor + position_enc
        return output_tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, query, key=None, value=None):
        batch_size = query.size(0)
        query = self.query(query)
        key = self.key(key) if key is not None else self.key(query)
        value = self.value(value) if value is not None else self.value(query)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        weights = torch.softmax(attn, dim=-1)
        out = torch.matmul(weights, value)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.fc(out)
        return out, weights


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, rate=0.3, layer_norm_eps=1e-5):
        super(FFN, self).__init__()

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(rate)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        y = self.linear2(self.dropout1(self.relu(self.linear1(x))))
        out = x + self.dropout2(y)
        out = self.norm(out)
        return out


# class SelfAtt(nn.Module):
#     def __init__(self, args):
#         super(SelfAtt, self).__init__()
#         self.att = MultiHeadAttention(args.d_model, args.num_heads)
#         self.layer_norm = nn.LayerNorm(args.d_model)
#         self.ffn = FFN(args.d_model, args.d_model * 2)

#         self.alpha = LearnableCoefficient()
#         self.beta = LearnableCoefficient()
#         self.gamma = LearnableCoefficient()
#         self.sigmma = LearnableCoefficient()

#     def forward(self, x):
#         out, _ = self.att(x)
#         out = self.layer_norm(self.alpha(x) + self.beta(out))
#         y = self.ffn(out)
#         out = self.layer_norm(self.gamma(out) + self.sigmma(y))
#         return out


# class Cross_Attention(nn.Module):
#     def __init__(self, args):
#         super(Cross_Attention, self).__init__()
#         self.att = MultiHeadAttention(args.d_model, args.num_heads)
#         self.layer_norm = nn.LayerNorm(args.d_model)
#         self.ffn = FFN(args.d_model, args.d_model * 2)

#         self.alpha = LearnableCoefficient()
#         self.beta = LearnableCoefficient()
#         self.gamma = LearnableCoefficient()
#         self.sigmma = LearnableCoefficient()

#     def forward(self, x, y):
#         out, _ = self.att(x, y, y)
#         z = self.layer_norm(self.alpha(x) + self.beta(out))
#         out = self.ffn(z)
#         out = self.layer_norm(self.gamma(out) + self.sigmma(z))
#         return out
    

class SelfAtt(nn.Module):
    def __init__(self, args):
        super(SelfAtt, self).__init__()
        self.att = MultiHeadAttention(args.d_model, args.num_heads)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.ffn = FFN(args.d_model, args.d_model * 2)

        self.alpha = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.sigmma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out, _ = self.att(x)
        out = self.layer_norm(x * self.alpha + out * self.beta)
        y = self.ffn(out)
        out = self.layer_norm(out * self.gamma + y * self.sigmma)
        return out


class Cross_Attention(nn.Module):
    def __init__(self, args):
        super(Cross_Attention, self).__init__()
        self.att = MultiHeadAttention(args.d_model, args.num_heads)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.ffn = FFN(args.d_model, args.d_model * 2)
        self.alpha = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.sigmma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x, y):
        out, _ = self.att(x, y, y)
        z = self.layer_norm(x * self.alpha + out * self.beta)
        out = self.ffn(z)
        out = self.layer_norm(out * self.gamma + z * self.sigmma)
        return out


class TimeAtt(nn.Module):
    def __init__(self, dims):
        super(TimeAtt, self).__init__()
        self.linear1 = nn.Linear(dims, dims, bias=False)
        self.linear2 = nn.Linear(dims, 1, bias=False)
        self.time = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        y = self.linear1(x.contiguous())
        y = self.linear2(torch.tanh(y))
        beta = F.softmax(y, dim=-1)
        c = beta * x
        return self.time(c.transpose(-1, -2)).transpose(-1, -2).contiguous().squeeze()


"""先Cross_Attention"""  # 走方案1
class Fusion_MMTM(nn.Module):
    def __init__(self, args):
        super(Fusion_MMTM, self).__init__()
        self.att1 = Cross_Attention(args)
        self.att2 = Cross_Attention(args)
        self.mmtm = NEW_MMTM2(args.d_model, args.d_model, args.dff, 4)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.dropout = nn.Dropout(args.drop_rate)
        self.fc = nn.Linear(args.d_model * 2, args.d_model)

        self.att3 = SelfAtt(args)
        self.fc1 = nn.Linear(args.d_model * 2, args.d_model)

    def forward(self, m1, m2, last=None):
        x = self.att1(m1, m2)  # 交叉注意力
        y = self.att2(m2, m1)
        x, y = self.mmtm(x, y)  # 给特征赋权值
        x = torch.cat((x, y), dim=-1)  # box与vel两支先进行融合
        x = self.dropout(self.fc(x))
        if last is not None:  # 进入第二次循环的时候，出现last
            z = self.att3(last)  # 将last进行自注意力
            x = self.dropout(self.fc1(torch.cat((x, z), dim=-1)))  # 与第一次融合的结果融合
        return x


# class MLP(nn.Module):  # 具有一个隐层的MLP
#     def __init__(self, dim, hidden_dim):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, dim)
#         self.gelu = nn.GELU()

#     def forward(self, x):
#         x = self.gelu(self.fc1(x))
#         x = self.fc2(x)
#         return x


"""new_MMTM2 D-MMTM架构"""
# class NEW_MMTM2(nn.Module):
#     def __init__(self, dim_bbox, dim_vel, hidden_dim, ratio):
#         super(NEW_MMTM2, self).__init__()
#         dim = dim_bbox + dim_vel
#         dim_out = int(2 * dim / ratio)
#         self.MLP = MLP(dim_out, hidden_dim)  # C->C    
#         self.encoding = nn.Sequential(nn.Linear(dim, 2*dim),   
#                                   nn.GELU(),
#                                   nn.Linear(2*dim,dim_out)
#                                   )  # 2C->C
#         self.decoding = nn.Sequential(nn.Linear(dim_out, 2*dim), 
#                                   nn.GELU(),
#                                   nn.Linear(2*dim,dim)
#                                   )  # C->2C
#         self.channel = dim_out
#         self.sigmoid = nn.Sigmoid()
#         self.maxpool = nn.AdaptiveMaxPool1d(1)
#         # LearnableCoefficient
#         self.E1_coefficient = LearnableWeight()
#         self.E2_coefficient = LearnableWeight()

#     def forward(self, bbox, vel):
#         S1_avg = self.MLP(torch.mean(bbox, dim=1))
#         S1_max = self.MLP(self.maxpool(bbox.transpose(-1, -2)).squeeze(-1))

#         S2_avg = self.MLP(torch.mean(vel, dim=1))
#         S2_max = self.MLP(self.maxpool(vel.transpose(-1, -2)).squeeze(-1))

#         # Concat avg & Concat max
#         Z_avg = torch.cat((S1_avg, S2_avg), 1)  # 将两支avg在第一维度Concat [B,2C]
#         Z_max = torch.cat((S1_max, S2_max), 1)  # 将两支max在第一维度Concat [B,2C]

#         # 过MLP Encoding  平均池化->encoding降维->激活函数->decoding升维
#         Z_avg = self.encoding(Z_avg)  # [B,C]
#         Z_max = self.encoding(Z_max)  # [B,C]
#         # 过MLP Decoding
#         Z_avg = self.decoding(Z_avg)  # [B,2C]
#         Z_max = self.decoding(Z_max)  # [B,2C]

#         # 拆分 & 按比例融合
#         E1_avg = Z_avg[:, :self.channel]
#         E1_max = Z_max[:, :self.channel]
#         Z_bbox = self.E1_coefficient(E1_avg, E1_max)  # [B,C]

#         E2_avg = Z_avg[:, self.channel:]
#         E2_max = Z_max[:, self.channel:]
#         Z_vel = self.E2_coefficient(E2_avg, E2_max)  # [B,C]

#         bbox_weight = self.sigmoid(Z_bbox)  # [B,C]
#         vel_weight = self.sigmoid(Z_vel)  # [B,C]

#         # 恢复形状
#         dim_diff = len(bbox.shape) - len(bbox_weight.shape)  # 3d-2d
#         bbox_weight = bbox_weight.view(bbox_weight.shape + (1,) * dim_diff).transpose(-1, -2)  # [B,C,1]->[B,1,C]
#         dim_diff = len(vel.shape) - len(vel_weight.shape)  # 3d-2d
#         vel_weight = vel_weight.view(vel_weight.shape + (1,) * dim_diff).transpose(-1, -2)  # [B,C,1]->[B,1,C]

#         return bbox * bbox_weight, vel * vel_weight  # [B,T,C]

class NEW_MMTM2(nn.Module):
    def __init__(self, dim_bbox, dim_vel, hidden_dim, ratio):
        super(NEW_MMTM2, self).__init__()
        dim = dim_bbox + dim_vel
        dim_out = int(2 * dim / ratio)
        self.MLP = nn.Linear(dim_out, dim_out)   # C->C    
        self.encoding = nn.Sequential(nn.Linear(dim, int(dim/2)),   
                                  nn.ReLU(),
                                  nn.Linear(int(dim/2), dim_out)
                                  )  # 2C->C
        self.decoding = nn.Sequential(nn.Linear(dim_out, dim), 
                                  nn.ReLU(),
                                  nn.Linear(dim, dim)
                                  )  # C->2C
        self.channel = dim_out
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        
    def forward(self, bbox, vel):
        """bbox [B,T,C]
           vel [B,T,C]"""
        # Squeeze
        # 对bbox,vel的时间维度求平均值[B,C]，求最大值[B,C]
        S1_avg = self.MLP(torch.mean(bbox, dim=1))
        S1_max = self.MLP(self.maxpool(bbox.transpose(-1, -2)).squeeze(-1))

        S2_avg = self.MLP(torch.mean(vel, dim=1))
        S2_max = self.MLP(self.maxpool(vel.transpose(-1, -2)).squeeze(-1))

        # Concat avg & Concat max
        Z_avg = torch.cat((S1_avg, S2_avg), 1)  # 将两支avg在第一维度Concat [B,2C]
        Z_max = torch.cat((S1_max, S2_max), 1)  # 将两支max在第一维度Concat [B,2C]

        # 过MLP Encoding  平均池化->encoding降维->激活函数->decoding升维
        Z_avg = self.encoding(Z_avg)  # [B,C]
        Z_max = self.encoding(Z_max)  # [B,C]
        # 过MLP Decoding
        Z_avg = self.decoding(Z_avg)  # [B,2C]
        Z_max = self.decoding(Z_max)  # [B,2C]

        # 拆分 & 按比例融合
        E1_avg = Z_avg[:, :self.channel]
        E1_max = Z_max[:, :self.channel]
        Z_bbox = self.alpha * E1_avg + (1-self.alpha) * E1_max # [B,C]

        E2_avg = Z_avg[:, self.channel:]
        E2_max = Z_max[:, self.channel:]
        Z_vel = self.beta * E2_avg + (1-self.beta) * E2_max # [B,C]

        bbox_weight = self.sigmoid(Z_bbox)  # [B,C]
        vel_weight = self.sigmoid(Z_vel)  # [B,C]

        dim_diff = len(bbox.shape) - len(bbox_weight.shape)  # 3d-2d
        bbox_weight = bbox_weight.view(bbox_weight.shape + (1,) * dim_diff).transpose(-1, -2)  # [B,C,1]->[B,1,C]
        dim_diff = len(vel.shape) - len(vel_weight.shape)  # 3d-2d
        vel_weight = vel_weight.view(vel_weight.shape + (1,) * dim_diff).transpose(-1, -2)  # [B,C,1]->[B,1,C]

        return bbox * bbox_weight, vel * vel_weight  # [B,T,C]
