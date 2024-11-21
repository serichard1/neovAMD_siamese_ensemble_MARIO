import torch
import torch.nn as nn
from torchvision.transforms import v2

class DualVision(nn.Module):
    def __init__(
        self,
        backbone,
        in_size,
        nclasses=4,
        drop_ratio_head=0.4
    ):
        super().__init__()

        self.backbone = backbone
        self.drop_ratio_head = drop_ratio_head

        self.backbone = backbone
        self.merge_bscans = self._create_sequential([in_size*2, 1024, 256, 32])
        self.merge_numeric = nn.Sequential(nn.Linear(32+3, 32), nn.SiLU(inplace=True))
        self.head =nn.Linear(32, nclasses)
        
    def _create_sequential(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=self.drop_ratio_head))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, bscan_ti, bscan_tj, bscan_num, age_ti, delta_t, localizer_ti=None):
        bscan_ti, bscan_tj = map(lambda f: self.backbone(f), (bscan_ti, bscan_tj))
        bscan_num, age_ti, delta_t = map(lambda f: f.unsqueeze(1), (bscan_num, age_ti, delta_t))

        bscans_embed = self.merge_bscans(torch.cat((bscan_ti, bscan_tj), dim=1))
        final_embed = self.merge_numeric(torch.cat((bscans_embed, bscan_num, age_ti, delta_t), dim=1))
        logits = self.head(final_embed)

        return logits
    
    
class CrossSightv5(nn.Module):
    def __init__(
        self,
        siamese_subnetworks,
        backbone,
        nclasses=3+1,
        dropout_head=0.3
    ):
        super().__init__()

        self.backbone = backbone
        
        self.model1, self.model2, self.model3 = siamese_subnetworks
        self.model1.head, self.model2.head, self.model3.head = nn.Identity(), nn.Identity(), nn.Identity()
        self.dropout_head = dropout_head

        self.backbone_head = self._create_sequential([1024, 256, 128, 64, 16])
        self.merging_embed = self._create_sequential([32*3+16, 96, 32, 16, nclasses])
        self.resize_incres = v2.Resize((256, 256), antialias=True)
        self.resize_effcnt = v2.Resize((224, 224), antialias=True)
        
    def _create_sequential(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=self.dropout_head))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, bscan_ti, bscan_tj, bscan_num, age_ti, delta_t, localizer_ti=None):
        embed1 = self.model1(bscan_ti, bscan_tj, bscan_num, age_ti, delta_t)
        embed2 = self.model2(self.resize_effcnt(bscan_ti), self.resize_effcnt(bscan_tj), bscan_num, age_ti, delta_t)
        embed3 = self.model3(self.resize_incres(bscan_ti), self.resize_incres(bscan_tj), bscan_num, age_ti, delta_t)

        embed_opt = self.backbone_head(self.backbone(localizer_ti))
        logits  = self.merging_embed(torch.cat((embed1, embed2, embed3, embed_opt), dim=1))

        return logits

