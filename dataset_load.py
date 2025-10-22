import os 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
import torchvision.transforms as transforms 

class ViTDataset(Dataset):
    def __init__(self, root, name_list, split, transform=None, target_transform=None, img_size=224):
        super().__init__()
        self.split = split 
        self.img_size = img_size  # 图像大小
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.target_transform = target_transform  # 标签变换
        # 构建数据集根目录
        self.data_dir = os.path.join(root, split)  # 训练集或测试集目录
        # 获取所有类别
        self.classes = name_list
        # self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # 收集所有图像文件路径和对应的标签
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                suff_exet = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')
                if img_name.lower().endswith(suff_exet):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        print(f"加载了 {len(self.images)} 张图像用于{split}集，共{len(self.classes)}个类别")
        print(self.class_to_idx)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # 获取图像路径和标签
        img_path = self.images[index]
        label = self.labels[index]
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 调整图像大小
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        # 应用变换
        image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return image, label
    

def ViTDataLoad(root, classes,batch_size, num_workers, img_size):
    # 创建训练数据集
    train_dataset = ViTDataset(
        root=root,
        name_list = classes,
        split='train',  # 使用训练集划分
        img_size=img_size
    )
    
    # 创建验证数据集
    val_dataset = ViTDataset(
        root=root,
        name_list=classes,
        split='val',  # 使用验证集划分
        img_size=img_size
    )
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 随机打乱数据
        num_workers=num_workers,  # 多线程加载
        pin_memory=True,  # 数据预加载到固定内存，加速GPU传输
        drop_last=True  # 丢弃最后不足一个批次的数据
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 不打乱数据
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
