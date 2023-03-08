from lib import *
from config import *
from utils import *
from image_transform import ImageTransform

class_index = ["ants", "bees"]

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index
    def predict_max(self, out_put):
        max_id = np.argmax(out_put.detach().numpy())
        predicted_label = self.class_index[max_id]
        return predicted_label
        
predictor = Predictor(class_index)        

def predict(img):
    # Tạo Mạng
    ### Prepare Network
    use_pretrained = True
    net = models.vgg16(pretrained = use_pretrained)
    #print(net)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.eval()
    # Prepare model
    model = load_model(net, save_path)
    # Prepare input img
    transform = ImageTransform(resize, mean, std)
    img = transform(img, phase='test')
    img = img.unsqueeze_(0)     #(chanel, height, width) => (1, chanel, height, width)
    ### PREDICT
    output = model(img)
    response = predictor.predict_max(output)
    
    return response