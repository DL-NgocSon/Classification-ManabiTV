from lib import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list, train_model, params_to_update, load_model
from dataset import MyDataset


def main():
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    ### DATASET ###
    train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase='train')
    val_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase='val')

    ### DATALOADER ###
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle = False
    )
    ### CREATE DICT ###
    # Tạo từ điển để lưu chữ cả 2 giúp dễ dàng sử dụng hơn
    dataloader_dict = {
        'train': train_dataloader,
        'val' : val_dataloader
    }

    ### NETWORK
    use_pretrained = True
    net = models.vgg16(pretrained = use_pretrained)
    net.classifier[6] = nn.Linear(in_features = 4096, out_features= 2)
    net.train()

    ### LOSS ###
    criterior = nn.CrossEntropyLoss()

    # ### update parameter ở tầng cuối cùng
    # params_to_update = []
    # update_params_name = ["classifier.6.weight", "classifier.6.bias"]
    # for name, param in net.named_parameters():
    #     if name in update_params_name:
    #         param.requires_grad = True
    #         params_to_update.append(param)
    #         print(param)
    #     else:
    #         param.requires_grad = False
    # print(params_to_update)

    params1, params2, params3 = params_to_update(net)
    ### OPTIMIZER
    optimizer = optim.SGD([
                            {'params': params1, 'lr': 1e-4},
                            {'params': params2, 'lr': 5e-4},
                            {'params': params3, 'lr': 1e-3},
                        ], momentum=0.9)
                # optimezer có tác dụng là khi mà các thông số parameters thay đổi thì => update thông số và tính lại

    # TRAINING
    train_model(net, dataloader_dict, criterior, optimizer, num_epochs)

if __name__ == "__main__":
    #main()
    use_pretrained = True
    net = models.vgg16(pretrained = use_pretrained)
    net.classifier[6] = nn.Linear(in_features = 4096, out_features= 2)
    net.train()
    load_model(net, save_path)