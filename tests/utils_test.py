import numpy as np
import utils
from utils import img_to_net, net_to_img





for i in range(100):
    net_data = np.random.rand(3, 10, 20)
    nd2 = img_to_net(net_to_img(net_data))
    assert np.all(net_data == nd2)

for i in range(100):
    img_data = np.random.rand(10, 20, 3)
    id2 = net_to_img(img_to_net(img_data))
    assert np.all(img_data == id2)
