import torch
try:
    from terratorch.models.backbones.terramind.model.terramind_register import v1_pretraining_mean, v1_pretraining_std
    terramind_mean = v1_pretraining_mean['untok_sen2l1c@224']
    terramind_std = v1_pretraining_std['untok_sen2l1c@224']
except ImportError:
    terramind_mean = None
    terramind_std = None

standardization_means = {
    'terramind': terramind_mean,
    'sen12mscrts': [3085.612548828, 2887.403320312, 2718.614746094, 2903.757568359, 3051.0546875, 3468.634765625, 3723.476074219, 3593.66015625, 3878.483154297, 1709.649047852, 314.885864258, 2729.378662109, 2093.306396484],
    'sen12mscrts_clip11k': [3085.319335938, 2886.961425781, 2718.328125, 2902.714355469, 3049.726318359, 3467.028320312, 3721.377197266, 3591.984375, 3876.041503906, 1709.608886719, 314.885864258, 2729.332275391, 2093.288085938],
    'sen12mscrts_clip10k': [3083.891357422, 2885.609619141, 2717.654052734, 2900.655761719, 3047.358398438, 3464.189941406, 3717.398925781, 3589.233886719, 3871.507568359, 1709.522949219, 314.885864258, 2729.273193359, 2093.275146484],
    'sen12mscrts_clip10k_RGB': [2900.655761719, 2717.654052734, 2885.609619141]
}

standardization_stds = {
    'terramind': terramind_std,
    'sen12mscrts': [2352.233642578, 2379.530029297, 2233.365234375, 2407.131835938, 2357.175537109, 2232.469482422, 2243.33203125, 2171.414306641, 2227.363037109, 1571.94921875, 696.844604492, 1387.53918457, 1170.120849609],
    'sen12mscrts_clip11k': [2351.146972656, 2377.851806641, 2232.147460938, 2403.066650391, 2351.965820312, 2226.10546875, 2235.273681641, 2164.639404297, 2217.976806641, 1571.690551758, 696.844604492, 1387.200073242, 1169.934814453],
    'sen12mscrts_clip10k': [2346.751708984, 2373.583740234, 2229.813964844, 2396.656982422, 2344.551025391, 2217.204833984, 2223.437500000, 2155.966796875, 2204.616455078, 1571.21105957, 696.844604492, 1386.881835938, 1169.828613281],
    'sen12mscrts_clip10k_RGB': [2396.656982422, 2229.813964844, 2373.583740234]
}


class ClipTransform():
    def __init__(self, clipVal):
        self.clipVal = clipVal

    def __call__(self, data: torch.Tensor):
        return torch.clamp(data, max=self.clipVal)


class NormalizationTransform():
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, data: torch.Tensor):
        return (data - self.min) / (self.max - self.min)


class DenormalizationTransform():
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, data: torch.Tensor):
        return data * (self.max - self.min) + self.min


class StandardizationTransform():
    def __init__(self, mean: list, std: list):
        self.mean = mean
        self.std = std

    def __call__(self, data: torch.Tensor):
        dev = data.device
        mean = torch.Tensor(self.mean).to(dev)
        std = torch.Tensor(self.std).to(dev)

        return (data - mean[None, :, None, None]) / std[None, :, None, None]


class DestandardizationTransform():
    def __init__(self, mean: list, std: list):
        self.mean = mean
        self.std = std

    def __call__(self, data: torch.Tensor):
        dev = data.device
        mean = torch.Tensor(self.mean).to(dev)
        std = torch.Tensor(self.std).to(dev)

        return data * std[None, :, None, None] + mean[None, :, None, None]


def standardizeTerramind(data):
    dev = data.device
    mean = torch.Tensor(standardization_means['terramind']).to(dev)
    std = torch.Tensor(standardization_stds['terramind']).to(dev)

    return (data - mean[None, :, None, None]) / std[None, :, None, None]


def destandardizeTerramind(data):
    dev = data.device
    mean = torch.Tensor(standardization_means['terramind']).to(dev)
    std = torch.Tensor(standardization_stds['terramind']).to(dev)

    return data * std[None, :, None, None] + mean[None, :, None, None]
