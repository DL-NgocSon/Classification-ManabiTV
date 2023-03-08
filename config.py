from lib import *

torch.manual_seed(2345)
np.random.seed(2345)
random.seed(2345)

num_epochs = 1
resize = 224
mean = (0.485, 0.456, 0.406) 
std = (0.229, 0.224, 0.225)

batch_size = 4  #Trong một lần training model sẽ train 2 ảnh


save_path = './weight_fine_tuning.pth'
