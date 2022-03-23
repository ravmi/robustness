

def net_to_img(data):
    assert data.ndim == 3
    return data.permute(1, 2, 0)

def img_to_net(data):
    assert data.ndim == 3
    return data.permute(2, 0, 1)
