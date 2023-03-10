from lib import *
from config import *

def make_datapath_list(phase='train'):
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath + phase + "/*/*.jpg")
    #print(target_path)
    
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
        
    return path_list


def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch, num_epochs))
        
        #move network to device(GPU/CPU)
        
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            if(epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                # Move inputs, label to GPU / CPU
                
                # Set gradient of optimizer to be zero
                optimizer.zero_grad()   #mỗi một epoch cần phải đưa giá trị về 0
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, axis = 1)
                    
                    
                    if phase == "train":
                        loss.backward()
                        
                        #update parameter trong optimizer
                        optimizer.step()

                    epoch_loss += loss.item()*inputs.size(0)   #lấy giá trị trong tensor
                    epoch_corrects += torch.sum(preds==labels)
                    
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            
            print("{} Loss: {:.4f} Accurancy: {:.4f}".format(phase, epoch_loss, epoch_accuracy))
            
    torch.save(net.state_dict(), save_path)

### Update các parameter của các lớp 1-4 và 5
def params_to_update(net):
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []
    
    update_param_name_1 = ["features"]
    update_param_name_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_name_3 = ["classifier.6.weight", "classifier.6.bias"]
    
    # Đưa vào network
    
    for name, param in net.named_parameters():
        if  name in update_param_name_1:
            param.requires_grad = True
            params_to_update_1.append(param)    #param tương ứng vs feature
        
        elif name in update_param_name_2:
            param.requires_grad = True
            params_to_update_2.append(param)
        
        elif name in update_param_name_3:
            param.requires_grad = True
            params_to_update_3.append(param)
        
        else:
            param.requires_grad = False
    return params_to_update_1, params_to_update_2, params_to_update_3

def load_model(net, model_path):
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights)
    print(net)
    
    for name, param in net.named_parameters():
        print(name, param)
    #load_weights = torch.load(model_path, map_location=("cuda:0", "cpu"))
    #net.load_state_dict(load_weights)