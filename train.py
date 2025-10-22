from model.transformer_net import VisonTransformer
from dataset_load import ViTDataLoad
import torch.optim.lr_scheduler as lr_scheduler
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

classes = ['1_残留物', '2_外延缺陷', '3_光刻图缺', '4_颗粒', '5_掉柱子',
           '6_圈状异常', '7_残缺管芯', "8_划伤", "9_刻蚀图缺", "10_栅氧damage"]
def train(
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
    epochs=100,
    batch_size=8,
    num_workers=4,
    lr=1e-3,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # 数据加载
    train_loader, val_loader = ViTDataLoad(root, classes, batch_size, num_workers, img_size)
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

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        cu_lr =optimizer.param_groups[0]['lr']  # 获取新的学习率
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Current lr={cu_lr:.10f}")
        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_vit.pth")

    # 可视化loss和acc
    plt.figure()
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")
    plt.figure()
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("acc_curve.png")
    print("训练完成，最优验证准确率：", best_acc)

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

    train()
