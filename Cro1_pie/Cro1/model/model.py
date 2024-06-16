import torch
from torch import nn
import torch.nn.functional as F
from model.model_blocks import PositionalEncoding, SelfAtt, TimeAtt, Fusion_MMTM


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv_bbox = nn.Sequential(nn.Conv1d(4, args.d_model, kernel_size=1, padding=0),
                                       nn.BatchNorm1d(args.d_model), nn.ReLU())
        self.conv_vel = nn.Sequential(nn.Conv1d(2, args.d_model, kernel_size=1, padding=0),
                                      nn.BatchNorm1d(args.d_model), nn.ReLU())
        self.positional_encoding = PositionalEncoding()

        """先Cross_Attention"""  #走方案2
        self.att_bbox = nn.ModuleList()
        self.att_vel = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for _ in range(args.num_layers):
            self.att_bbox.append(SelfAtt(args))
            self.att_vel.append(SelfAtt(args))
            self.fusion.append(Fusion_MMTM(args))

        self.num_layers = args.num_layers

        self.fc_bbox = nn.Linear(args.d_model, args.num_class)
        self.time_bbox = TimeAtt(args.d_model)
        self.fc_vel = nn.Linear(args.d_model, args.num_class)
        self.time_vel = TimeAtt(args.d_model)
        self.fc_fusion = nn.Linear(args.d_model, args.num_class * 4)
        self.time_fusion = TimeAtt(args.d_model)

        self.last = nn.Linear(args.num_class * 4, args.num_class)
        self.softplus = nn.Softplus()

        self.params = nn.Parameter(torch.randn(1, 3, requires_grad=True), requires_grad=True)

    """先Cross_Attention"""  #走方案2
    def forward(self, bbox, vel):
        # bbox:[B,C,T]
        # vel:[B,C,T]
        bbox = self.positional_encoding(self.conv_bbox(bbox)).transpose(-1, -2)  # B,T,C
        vel = self.positional_encoding(self.conv_vel(vel)).transpose(-1, -2)  # B,T,C

        fusion = None
        for i in range(self.num_layers):
            bbox = self.att_bbox[i](bbox)  # B,T,C
            vel = self.att_vel[i](vel)  # B,T,C
            fusion = self.fusion[i](bbox, vel, last=fusion)  # B,T,C

        pred_bbox = self.fc_bbox(self.time_bbox(bbox))  # B,2
        pred_vel = self.fc_vel(self.time_vel(vel))  # B,2
        pred_fusion = self.last(self.fc_fusion(self.time_fusion(fusion)))  # B,2

        return self.softplus(pred_bbox), self.softplus(pred_vel), self.softplus(pred_fusion)
