from model.transformer_net import VisonTransformer
from dataset_load import ViTDataLoad
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

def export_onnx(weight_path,
    root="F:/AlgoData/SN003/images_clsdata", # 数据集根目录
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_features=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1,
    batch_size=4,
    num_workers=4,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # # 数据加载
    train_loader, val_loader = ViTDataLoad(root, batch_size, num_workers, img_size)
    num_classes = len(train_loader.dataset.classes)
    input_shape = (in_channels, img_size, img_size)

    # 模型
    model = VisonTransformer(
        input_shape=input_shape,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        num_features=num_features,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU
    ).to(device)

    params_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(params_dict)
    model.eval()
    # 导出模型
    dummy_input = torch.randn(1, in_channels, img_size, img_size).to('cpu')
    model.to('cpu')
    torch.onnx.export(
        model,  # PyTorch 模型
        dummy_input,  # 示例输入
        'best_vit.onnx',
        # export_params=True,  # 是否导出模型参数
        opset_version=11,  # ONNX 的 opset 版本
        # do_constant_folding=True,  # 是否进行常量折叠优化
        input_names=['input'],  # 输入的名称
        output_names=['output'],  # 多个输出的名称
    )


if __name__ == "__main__":
    weight_path = 'best_vit.pth'
    export_onnx(weight_path)
