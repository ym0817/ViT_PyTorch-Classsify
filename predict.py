import shutil
import torch
from model.transformer_net import VisonTransformer
import torchvision.transforms as transforms
from PIL import Image
import sys,os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


img_size = 224
patch_size = 16
in_channels = 3
num_features = 768
depth = 12
num_heads = 12
mlp_ratio = 4.0
qkv_bias = True
drop_rate = 0.1
attn_drop_rate = 0.1
drop_path_rate = 0.1

# classes = ['cat', 'dog']
classes = ['1_残留物', '2_外延缺陷', '3_光刻图缺', '4_颗粒', '5_掉柱子',
'6_圈状异常', '7_残缺管芯', "8_划伤", "9_刻蚀图缺", "10_栅氧damage"]
num_classes = len(classes)
input_shape = (in_channels, img_size, img_size)

def load_model(device):
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
        norm_layer=torch.nn.LayerNorm,
        act_layer=torch.nn.GELU
    ).to(device)
    model.load_state_dict(torch.load("best_vit.pth", map_location=device))
    model.eval()
    return model

def predict_one(img_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()
    return classes[pred]


def plot_confusion_matrix(y_true, y_pred, classes, result_dir,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if normalize:
        plt.savefig(os.path.join(result_dir,'normalized-result.png'))
    else:
        plt.savefig(os.path.join(result_dir,'non-normalized-result.png'))
    return ax


def inference_files(model,label_names, image_folder_path, result_dir ):

    suff_exet = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')
    all_preds = []
    all_labels = []
    for dirpath, _, filenames in os.walk(image_folder_path):
        for fname in filenames:
            if fname.lower().endswith(suff_exet):
                file_path = os.path.join(dirpath, fname)
                label_name = os.path.basename(os.path.split(file_path)[0])
                if label_name in label_names:
                    label_index = label_names.index(label_name)
                    # img = Image.open(file_path)  # .convert('L')  # Grayscale
                    # img_tensor = transform(img).unsqueeze(0).to(device)
                    # # print("img_tensor , ", img_tensor.shape)
                    # outputs = model(img_tensor)

                    pred_name = predict_one(file_path, model, device)
                    pred_index = classes.index(pred_name)

                    # _, preds = torch.max(outputs, 1)
                    # pred_index = preds.cpu().numpy()
                    all_preds.append(pred_index)
                    all_labels.append(label_index)
                    save_dir = os.path.join(result_dir,label_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    new_name =  label_name + "识别为_" + pred_name + '_' + fname
                    shutil.copy(file_path, os.path.join(save_dir, new_name))


                else:
                    print("{label_name} is not in {label_names}")
        # Plot non-normalized confusion matrix
    ax1 = plot_confusion_matrix(all_labels, all_preds, classes, result_dir,
                                title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    ax2 = plot_confusion_matrix(all_labels, all_preds, classes, result_dir, normalize=True,
                                title='Normalized confusion matrix')

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    # 推理单张图片
    # img_path = sys.argv[1]
    # if not os.path.exists(img_path):
    #     print(f"图片不存在: {img_path}")
    #     sys.exit(1)
    # pred_class = predict_one(img_path, model, device)
    # print(f"图片 {img_path} 的预测类别为: {pred_class}")

    # 推理整个文件夹
    image_folder_path = "F:/AlgoData/SN003/images_clsdata/val"
    results = image_folder_path + "_res1022"
    inference_files(model, classes, image_folder_path,results)
