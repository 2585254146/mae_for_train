import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch import nn

import config


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))  # remain 是输入encoder的token数

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=(224, 224),
                 image_channels=3,
                 patch_size=(16, 16),
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size[0])*(image_size[1] // patch_size[1]), 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(in_channels=image_channels, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)

        self.transformer = torch.nn.Sequential(*[Block(dim=emb_dim, num_heads=num_head, drop_path=0.1) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=(224, 224),
                 patch_size=(16, 32),
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))   # mask_token is learning
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(in_features=emb_dim, out_features=1 * patch_size[0]*patch_size[1])
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=image_size[0] // patch_size[0], w=image_size[1] // patch_size[1])

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]  # T is the numbers of Tokens
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 img_size=(224, 224),
                 in_chans=3,
                 patch_size=(16, 16),
                 encoder_emb_dim=768,  # 192
                 encoder_layer=3,
                 encoder_head=12,
                 decoder_emb_dim=128,
                 decoder_layer=1,
                 decoder_head=2,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(img_size, in_chans, patch_size, encoder_emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(img_size, patch_size, decoder_emb_dim, decoder_layer, decoder_head)
        self.encoder_2_decoder_proj = nn.Linear(encoder_emb_dim, decoder_emb_dim)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        # project encoder embeeding to decoder embeeding
        features = self.encoder_2_decoder_proj(features)
        predicted_img, mask = self.decoder(features,  backward_indexes)

        return {"predicted_img":  predicted_img,
                "mask_matrix":     mask}

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=50) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


def get_model(model_config: dict, weights_path: str, mode:str, load_pretrain:str, Test:str):
    if mode == "ssl":
        model = MAE_ViT(**model_config)
        if load_pretrain == True:
            state_dict = torch.load(weights_path, map_location=config.sys_config["device"])
            model.load_state_dict(state_dict['model_state_dict'], strict=True)
    elif mode == "ds":
        MAE = MAE_ViT(**model_config)
        if load_pretrain == True:
            state_dict = torch.load(weights_path, map_location=config.sys_config["device"])
            MAE.load_state_dict(state_dict['model_state_dict'], strict=True)
        model = ViT_Classifier(MAE.encoder, num_classes=config.Data.NUM_CLASSES)
    elif mode == "sl":
        MAE = MAE_ViT(**model_config)
        model = ViT_Classifier(MAE.encoder)
        if load_pretrain == True:
            state_dict = torch.load(weights_path, map_location=config.sys_config["device"])
            model.load_state_dict(state_dict['model_state_dict'], strict=True)

    if Test == True:
        model.eval()
    else:
        model.train()

    model.to(config.sys_config["device"])

    return model

if __name__ == '__main__':
    # shuffle = PatchShuffle(0.75)
    # a = torch.rand(16, 2, 10)
    # b, forward_indexes, backward_indexes = shuffle(a)
    # print(b.shape)
    #
    # img = torch.rand(2, 3, 32, 32)
    # encoder = MAE_Encoder()
    # decoder = MAE_Decoder()
    # features, backward_indexes = encoder(img)
    # print(forward_indexes.shape)
    # predicted_img, mask = decoder(features, backward_indexes)
    # print(predicted_img.shape)
    # loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    # print(loss)
    img = torch.rand(2, 1, 128, 256)
    model = MAE_ViT()
    output = model(img)
    loss = torch.mean((output["predicted_img"] - img) ** 2 * output["mask_matrix"] / 0.75)
    print(loss)