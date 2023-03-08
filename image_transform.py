from lib import *

class ImageTransform():
  def __init__(self, resize, mean, std):
    self.data_transform = {
        #Tiền xử lý ảnh cho training
        'train' : transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)), #random nhằm tạo ra nhiều dạng data => quá trình học hiệu quả hơn.
            transforms.RandomHorizontalFlip(0.7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        #Tiền xử lý ảnh cho validation
        'val' : transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test' : transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
  #call sẽ tự động được gọi khi gọi đến class mẹ
  def __call__(self, img, phase='train'): #mode train/vali
    return self.data_transform[phase](img)
