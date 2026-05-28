import torch
import torch.nn as nn

class RandomizedQuantizationAugModule(nn.Module):
    def __init__(self, region_num, collapse_to_val = 'inside_random', spacing='random', transforms_like=False, p_random_apply_rand_quant = 1):
        """region_num: 分箱数 int。"""
        super().__init__()
        self.region_num = region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.p_random_apply_rand_quant = p_random_apply_rand_quant

    def get_params(self, x):
        """x: (C, H, W); 返回每通道的 min/max/分位点数。"""
        C, _, _ = x.size()
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0]
        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int()
        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """x: (B, c, H, W) or (C, H, W)。"""
        EPSILON = 1
        if self.p_random_apply_rand_quant != 1:
            x_orig = x
        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c
            x = x.view(C, H, W)
        else:
            C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x)

        # 每通道随机/均匀分位点
        if self.spacing == "random":
            region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)
        elif self.spacing == "uniform":
            region_percentiles = torch.tile(torch.arange(1/(total_region_percentile_number_per_channel[0] + 1), 1, step=1/(total_region_percentile_number_per_channel[0]+1), device=x.device), [C])
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])
        # 排序后的分箱左右端点
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1, 1)
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]
        # 分箱中点
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2

        # 确定每像素所属分箱 id
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.view(C, 1, H, W) >= ordered_region_left_ends)
        assert (is_inside_each_region.sum(1) == 1).all()
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)

        if self.collapse_to_val == 'middle':
            # 用分箱中点作为代理值
            proxy_vals = torch.gather(ordered_region_mid.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'inside_random':
            # 在所属分箱内均匀采样一个代理值
            proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(), device=x.device)
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, self.region_num])
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (ordered_region_right_ends - ordered_region_left_ends)
            proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'all_zeros':
            proxy_vals = torch.zeros_like(x, device=x.device)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
            x = x.view(B, c, H, W)

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                x = torch.where(torch.rand([B,1,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                x = torch.where(torch.rand([C,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)

        return x
