

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import copy
from torchvision.models import wide_resnet50_2
import common  




class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension

    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(3, stride=1)
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator


    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]


        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        '''
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

'''

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        #features = self.forward_modules["preprocessing"](features)
        #features = self.forward_modules["preadapt_aggregator"](features)

        f2 = features[0]  # لایه دوم
        f3 = features[1]  # لایه سوم
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([f2, f3_up], dim=1)
        features = [fused]
        features = (features)[0].squeeze(0)


        #if provide_patch_shapes:
            #return _detach(features), patch_shapes
            #return _detach(features), None
        #return _detach(features)[0]
        return (features.detach().cpu().numpy())


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return


class LastLayerToExtractReachedException(Exception):
    pass


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x


#@@@@@@@@@@@@@@@@@@@@@@@@@@


# مسیرها
dataset_root = "./mvtec_original"
features_save_root = "./features_test_pixel"
os.makedirs(features_save_root, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# مدل
model = PatchCore(device)
backbone = wide_resnet50_2(weights='IMAGENET1K_V1')
layers_to_extract_from = ["layer1", "layer2"]
input_shape = (3, 224, 224)
model.load(
    backbone=backbone,
    layers_to_extract_from=layers_to_extract_from,
    device=device,
    input_shape=input_shape,
    pretrain_embed_dimension=1024,
    target_embed_dimension=1024,
)

# transformها
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
transform_gt = transforms.Compose([
    transforms.Resize((56, 56)),  # چون feature map هم 56x56 است
    transforms.ToTensor(),
])

# پردازش داده‌ها
for class_name in os.listdir(dataset_root):
    class_test_dir = os.path.join(dataset_root, class_name, "test")
    if not os.path.isdir(class_test_dir):
        continue

    class_features_dir = os.path.join(features_save_root, class_name)
    os.makedirs(class_features_dir, exist_ok=True)

    gt_root = os.path.join(dataset_root, class_name, "ground_truth")

    # --- تصاویر نرمال (good) ---
    good_dir = os.path.join(class_test_dir, "good")
    if os.path.isdir(good_dir):
        normal_save_dir = os.path.join(class_features_dir, "normal")
        os.makedirs(normal_save_dir, exist_ok=True)

        good_images = [f for f in os.listdir(good_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"Found {len(good_images)} normal (good) images in {class_name}")

        for img_name in tqdm(good_images, desc=f"{class_name}/good"):
            img_path = os.path.join(good_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform_img(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.embed(img_tensor)  # (768, 56, 56)

            np.save(os.path.join(normal_save_dir, os.path.splitext(img_name)[0] + ".npy"), features)

    # --- تصاویر غیرنرمال ---
    anomaly_types = [
        d for d in os.listdir(class_test_dir)
        if d != "good" and os.path.isdir(os.path.join(class_test_dir, d))
    ]

    for anomaly_type in anomaly_types:
        anomaly_dir = os.path.join(class_test_dir, anomaly_type)

        anomaly_save_dir = os.path.join(class_features_dir, "anomaly")
        os.makedirs(anomaly_save_dir, exist_ok=True)

        image_files = [f for f in os.listdir(anomaly_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for img_name in tqdm(image_files, desc=f"{class_name}/{anomaly_type}"):
            img_path = os.path.join(anomaly_dir, img_name)
            gt_path = os.path.join(gt_root, anomaly_type, img_name[:-4] + "_mask.png")  # الگوی نام MVTec

            if not os.path.exists(gt_path):
                continue

            img = Image.open(img_path).convert("RGB")
            gt = Image.open(gt_path).convert("L")

            img_tensor = transform_img(img).unsqueeze(0).to(device)
            gt_tensor = transform_gt(gt).squeeze(0)  # (56,56)

            with torch.no_grad():
                features = model.embed(img_tensor)

            # تقسیم ویژگی‌ها بر اساس ground truth
            normal_feats = []
            anomaly_feats = []

            for i in range(56):
                for j in range(56):
                    pixel_feat = features[:, i, j]  # (768,)
                    if gt_tensor[i, j] >= 0.5:  # پیکسل ناهنجار
                        anomaly_feats.append(pixel_feat)
                    else:  # پیکسل نرمال
                        normal_feats.append(pixel_feat)

            normal_feats = np.stack(normal_feats) if len(normal_feats) > 0 else np.empty((0, 768))
            anomaly_feats = np.stack(anomaly_feats) if len(anomaly_feats) > 0 else np.empty((0, 768))

            np.save(os.path.join(anomaly_save_dir, f"{anomaly_type}_{os.path.splitext(img_name)[0]}_normal.npy"), normal_feats)
            np.save(os.path.join(anomaly_save_dir, f"{anomaly_type}_{os.path.splitext(img_name)[0]}_anomaly.npy"), anomaly_feats)
