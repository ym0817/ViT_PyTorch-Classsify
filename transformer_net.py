import torch
from torch import nn
from torch.nn import functional as F

class VisionPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, flatter=True):
        super().__init__()
        self.image_size = image_size # 输入图片大小
        self.patch_size = patch_size # 一张图片分割出的patch大小
        self.in_channels = in_channels # 输入图片的通道数
        self.embed_dim = embed_dim # 嵌入向量维度 Vit_Base中为768 Vit_Small = 384 Vit_Large = 1024
        self.flatter = flatter # 是否将patch展平

        self.proj = nn.Conv2d(self.in_channels, self.embed_dim, self.patch_size, self.patch_size) # 通过卷积中卷积核与步长相等来将图片划分为 Patch * Patch * Channels 大小
        self.norm = nn.LayerNorm(self.embed_dim) # 对每个patch进行归一化
        
    def forward(self, x):
        x = self.proj(x)
        if self.flatter:
            x = x.flatten(2).transpose(1,2) # BCHW -> BNC N:Patch**2
        x = self.norm(x)
        return x
       
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop_rate=0.0, proj_drop_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn .softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer, drop_rate):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop_rate, drop_rate)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob, training):
        if drop_prob == 0. or not training:
            return x
        keep_prob       = 1 - drop_prob
        shape           = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor   = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() 
        output          = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_radio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer):
        super().__init__()
        self.norm_1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop_rate=attn_drop, proj_drop_rate=drop)
        self.norm_2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_radio), out_features=None, act_layer=act_layer, drop_rate=drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity() # 丢弃路径

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm_1(x)))
        x = x + self.drop_path(self.mlp(self.norm_2(x)))
        return x
    
class VisonTransformer(nn.Module):
    def __init__(self, input_shape, patch_size, in_channels, num_classes, num_features, depth,
                 num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, act_layer):
        super().__init__()
        self.input_shape = input_shape # 输入的维度
        self.patch_size = patch_size # Patch 的大小
        self.in_channels = in_channels # 输入的维度
        self.num_classes = num_classes # 输出类别数
        self.num_features = num_features # 特征维度
        self.depth = depth # Transformer编码器层数
        self.num_heads = num_heads # Transformer注意力头数
        self.mlp_ratio = mlp_ratio # MLP 比例 MLP:多层感知机,紧随 Self-Attention 之后，用于非线性变换：增强模型的表达能力；特征映射：将 Self-Attention 提取的特征进一步转换。 
        self.qkv_bias = qkv_bias # 是否使用偏置
        self.drop_rate = drop_rate # 丢弃率
        self.attn_drop_rate = attn_drop_rate # 注意力丢弃率
        self.drop_path_rate = drop_path_rate # 丢弃路径率
        self.norm_layer = norm_layer # 归一化层
        self.act_layer = act_layer # 激活函数层

        self.features_shape = [input_shape[1] // patch_size, input_shape[2] // patch_size]  # [14, 14]
        self.num_patches = self.features_shape[0] * self.features_shape[1]
        self.patch_embed = VisionPatchEmbedding(input_shape, patch_size, in_channels, num_features) # 将输入图片分割成patch，并进行线性映射

        # ViT 不是 CNN，没有"感受野"，所以引入了位置编码，来为每个 patch 加上位置信息；
        self.pretrained_features_shape = [224 // patch_size, 224 // patch_size] # 预训练的特征图尺寸

        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features)) # 分类 token 196, 768 -> 197, 768
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features)) # 位置编码 197, 768 -> 197, 768

        self.pos_drop = nn.Dropout(drop_rate) # 丢弃率
        self.norm = norm_layer(self.num_features) # 归一化

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # 丢弃路径率
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim = num_features,
                    num_heads = num_heads,
                    mlp_radio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    drop = drop_rate,
                    attn_drop = attn_drop_rate,
                    drop_path = self.dpr[i],
                    norm_layer = norm_layer,
                    act_layer = act_layer 
                )for i in range(depth)
            ]
        )
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self,x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # 将分类 token 扩展到与输入特征图相同的形状
        x = torch.cat((cls_token, x), dim=1) # 将分类 token 与输入特征图拼接

        cls_token_pos_embed = self.pos_embed[:, 0:1, :] # 分类 token 的位置编码
        img_token_pos_embed = self.pos_embed[:, 1:, :]  # [1, num_patches, num_features]
        # 变成[1, H, W, C]
        img_token_pos_embed = img_token_pos_embed.view(1, self.features_shape[0], self.features_shape[1], -1).permute(0, 3, 1, 2)  # [1, C, H, W]
        # 插值
        img_token_pos_embed = F.interpolate(
            img_token_pos_embed,
            size=self.features_shape,  # [H, W]
            mode='bicubic',
            align_corners=False
        )
        # 变回[1, num_patches, C]
        img_token_pos_embed = img_token_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, img_token_pos_embed.shape[1])

        pos_embed = torch.cat((cls_token_pos_embed, img_token_pos_embed), dim=1) # 将分类 token 的位置编码与图像 token 的位置编码拼接
        
        x = self.pos_drop(x + pos_embed) # 将位置编码与输入特征图相加

        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0] # 返回分类 token 的特征
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    
