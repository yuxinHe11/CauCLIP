import pdb
from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

DWCONV3D_DISABLE_CUDNN = True

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# Used for t-adapter
class Adapter1(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        # kernel_size = 3
        # T = num_frames
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        # x: Tensor: [bs*num_frames, 512]
        T = self.T
        BT, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        x_id = x
        x = self.fc1(x)
        x = x.view(B, T, Ca).permute(0, 2, 1).contiguous().view(B,Ca,T)  #

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(-1,Ca)
        x = self.fc2(x)
        x_id = x + x_id # x_id: Tensor: [bs*num_frames, 512]
        return x_id

# Used for adapter in vision transformer
class Adapter2(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv1d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T # num_frames
        self.offset_adapter = OffsetAdapter(in_channels,adapter_channels,(1,3,3),T)
        self.offset_adapter = self.offset_adapter.half()
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        offset = self.offset_adapter(x) # [BT, L - 1, C1] 偏移量
        T = self.T
        BT, L, C = x.size()             # [BT, L, C]
        B = BT // T
        C1 = self.conv.in_channels      # adapter_channels e.g. 384
        x_id = x
        x = self.fc1(x)                 # [BT, L, C1]
        x = x.view(B, T, L, C1).permute(0, 2, 3, 1).contiguous().view(B*L, C1, T) # [BL, C1, T]

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x) # 在最后一个维度T上进行一维卷积
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 1).contiguous().view(B, L, T, C1)
        x = x.permute(0, 2, 1, 3).contiguous().view(BT, L, C1) # [BT, L, C1]
        x = self.fc2(x) # [BT, L, C]
        x_id = x_id.clone()
        x_id[:, 1:, :] = x_id[:, 1:, :] + offset
        x_id = x + x_id
        
        return x_id

# Used in Adapter2
class OffsetAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, T):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        self.T = T
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = self.T
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W

        x = x[:, 1:, :].view(B, T, -1, C) # 二维卷积不对CLS token做
        former_id = [0] + [i for i in range(T)][:-1]
        x_former = x[:,former_id]

        offset = x - x_former
        offset = self.fc1(offset)
        offset = offset.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()
        
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        offset = offset.to(self.conv.weight.dtype)
        offset = self.conv(offset)
        torch.backends.cudnn.enabled = cudnn_enabled

        offset = offset.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        offset = offset.to(self.fc2.weight.dtype)
        offset = self.fc2(offset)
        return offset
    

class ResidualAttentionBlock_adapter(nn.Module):
    '''
    Implemention of Residual Attention Block with Adapter.
    '''
    def __init__(
            self, 
            d_model: int, 
            n_head: int, 
            attn_mask: torch.Tensor = None,
            adapter_width:int = 384,
            dropout = 0.,
            num_frames = 16
        ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)     
        self.ln_1 = LayerNorm(d_model)                                               
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()   
        self.mlp = nn.Sequential(OrderedDict([                                  
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.T = num_frames
        self.attn_mask = attn_mask

        self.adapter_pre_mlp = Adapter2(
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size=3,
            T=self.T
        )

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = x.permute(1, 0, 2)
        x = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, average_attn_weights=True)[0]
        x = x.permute(1, 0, 2) # [bs*num_frames, 197, 768]?
        return x

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = self.adapter_pre_mlp(x)
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock(nn.Module):
    '''
    Original implemention of Residual Attention Block.
    '''
    def __init__(
            self, 
            d_model: int, 
            n_head: int, 
            attn_mask: torch.Tensor = None, 
            dropout = 0.,
        ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)     
        self.ln_1 = LayerNorm(d_model)                                                 
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()   
        self.mlp = nn.Sequential(OrderedDict([                                  
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None # literally None
        ret = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, average_attn_weights=True)          # need_weights=False <=> len=1 else len=2
        return ret[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None, adp:bool=False, num_frames:int=16):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for _ in range(layers)]
        print(f'Initializing transformer:\ntotal layers:{layers}\ndropout used:{dropout}\nwith adapter:{adp}\n------------------------')
        self.width = width
        self.layers = layers
        self.with_adapter = adp
        if not self.with_adapter:
            # w/o adapter
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock(
                    d_model=width, 
                    n_head=heads, 
                    attn_mask=attn_mask, 
                    dropout=dropout[i]) for i in range(layers)]
            )
        else:
            # w/ adapter
            self.resblocks = nn.ModuleList([
                ResidualAttentionBlock_adapter(
                    d_model=width,
                    n_head=heads,
                    attn_mask=attn_mask,
                    adapter_width=384,
                    dropout=0,
                    num_frames=num_frames
                ) for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for i, block in enumerate(self.resblocks):
            x = block(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)   
        )

    def forward(self, x):
        x = self.ffn(x)        
        x = x.mean(dim=1)                    
        return x


class VisualTransformer(nn.Module):
    def __init__(
            self, 
            input_resolution: int, 
            patch_size: int, 
            width: int, 
            layers: int, 
            heads: int, 
            output_dim: int,
            dropout = None,
            joint=False, 
            emb_dropout = 0.,
            T=None, 
            add_channel=False,
            adp=False,
            num_frames=16,
            return_patch_token=True
        ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        if not add_channel:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
            print('Conv2D with in_channels = 3.\n------------------------')
        else:
            self.conv1 = nn.Conv2d(in_channels=4, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
            print('Conv2D with in_channels = 4.\n------------------------')

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        self.joint = joint
        self.adp = adp
        self.num_frames = num_frames
        self.return_patch_token = return_patch_token

        if not self.adp:
            print('ViT w/o adapter!\n------------------------')

        if emb_dropout > 0:
            print('emb_dropout:{}'.format(emb_dropout))

        if self.return_patch_token:
            print('ViT output patch token!\n------------------------')
        else:
            print('ViT output CLS token!\n------------------------')

        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, dropout=dropout, adp=self.adp, num_frames=num_frames)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.patchMLP = MLP(512, 2048, 512)
        self.MLP = MLP(1024, 2048, 512)

    def forward(self, x: torch.Tensor):
        # bs = train batch size * num_segments(num_frames)
        # x: [bs, channels, 224, 224]
        x = self.conv1(x)                   # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)              # shape = [*, grid ** 2, width]
                                            # [bs, 196, width] -> [bs, 197, width] CLS加在seq的头部
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype) 
        x = self.ln_pre(x)                  # [bs, 197, 768]

        if not self.adp:
            x = x.permute(1, 0, 2)          # [197, bs, 768] 若Transformer的初始化参数adp=False，2次更换维度不能省略
        x = self.transformer(x)
        if not self.adp:
            x = x.permute(1, 0, 2)          # [bs, 197, 768]

        x1 = self.ln_post(x)                # [bs, 197, 768]
        x1 = x1 @ self.proj                 # [bs, 197, 512]
        patch_token = x1[:, 1:, :]          # [bs, 196, 512]
        CLS_token = x1[:, 0, :]             # [bs, 512]

        if self.return_patch_token:
            patch_token = self.patchMLP(patch_token)                         
            patch_token = patch_token / patch_token.norm(dim=-1, keepdim=True)
            CLS_token = CLS_token / CLS_token.norm(dim=-1, keepdim=True)
            CLS_token = torch.cat([CLS_token, patch_token], dim=-1) 
            CLS_token = self.MLP(CLS_token)

        return CLS_token                                                 


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 add_channel:bool,
                 adp:bool,
                 num_frames:int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 joint=False,
                 tsm=False, 
                 T=8,
                 dropout=0., 
                 emb_dropout=0.,
                 visual_patch_token=False
                ):
        super().__init__()

        self.context_length = context_length
        if dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, dropout, vision_layers)]  # stochastic depth decay rule
        else:
            dpr = None

        # embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768,  vision_patch_size=16
        # context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,joint=joint,dropout=dpr,
            emb_dropout=emb_dropout,
            add_channel=add_channel,
            adp=adp,
            num_frames=num_frames,
            return_patch_token=visual_patch_token
        )
        if tsm:
            print('=========using TSM==========')
            from modules.temporal_shift import make_temporal_shift_vit
            make_temporal_shift_vit(self.visual, T)

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout=dpr
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        
        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx=77, transformer.width=512], text.shape=[bs, 77]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # [bs, 512]

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)  #  [bs, 512]
        text_features = self.encode_text(text)  #  [bs, 512]

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(
        state_dict: dict, 
        tsm=False, 
        T=8, 
        dropout=0., 
        joint=False, 
        emb_dropout=0., 
        pretrain=True, 
        add_channel:bool=False, 
        adp:bool=False, 
        num_frames:int=16,
        visual_patch_token=False
    ):
    vit = "visual.proj" in state_dict
    if add_channel: # add_channel=True, add a input channel in conv2D
        print('Add a channel in Conv2D!')
        if state_dict["visual.conv1.weight"].shape[1] == 3:
            print("pretrained conv1 does not have 4 input channels, initialize.")
            old_w = state_dict["visual.conv1.weight"]  # shape: [768, 3, 16, 16]
            new_w = torch.zeros(old_w.shape[0], 4, old_w.shape[2], old_w.shape[3])  # shape: [768, 4, 16, 16]
            new_w[:, :3] = old_w
            new_w[:, 3] = old_w.mean(dim=1)  # 初始化卷积层第4通道权重，采用前3通道均值。
            state_dict["visual.conv1.weight"] = new_w
        else:
            print("pretrained conv1 does have 4 input channels.")

    print(f'visual.conv1.weight:\n{state_dict["visual.conv1.weight"].size()}') # [768, C, 16, 16]

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    # pdb.set_trace()
    # embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768,  vision_patch_size=16
    # context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12
    model = CLIP(
        embed_dim,
        image_resolution, 
        vision_layers, 
        vision_width, 
        vision_patch_size,
        add_channel,
        adp,
        num_frames,
        context_length, 
        vocab_size, 
        transformer_width, 
        transformer_heads, 
        transformer_layers,  
        tsm=tsm,
        T=T,
        joint=joint,
        dropout=dropout, 
        emb_dropout=emb_dropout,
        visual_patch_token=visual_patch_token
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    if tsm:
        for k in list(state_dict.keys()):
            if k.find("conv1")>-1 and k.find("layer")>-1: 
                n_k = k.split('conv1.')[0]+'conv1.net.'+k.split('conv1.')[1]
                state_dict[n_k] = state_dict.pop(k)
            if k.find("resblocks")>-1 and k.find("visual")>-1: 
                tmp = ''
                for i, t_ in enumerate(k.split('resblocks.')[1].split('.')):
                    if i>=1:
                        tmp += '.' + t_ 
                
                n_k = k.split('resblocks.')[0]+'resblocks.' + k.split('resblocks.')[1].split('.')[0]+'.net'+ tmp
#                 print(n_k)
                state_dict[n_k] = state_dict.pop(k)

    convert_weights(model)
    if pretrain:
        print('loading clip pretrained model!')
        if joint:  #or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict, strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
            # print("from model.py, Missing keys:", msg.missing_keys)
    else:
        print('not using full clip pretrained model, only visual!')
        
        for k in list(state_dict.keys()):
            if not k.find("visual")>-1: 
                state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)

    return model.eval()