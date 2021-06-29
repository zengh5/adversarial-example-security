
"""
 A demo for detecting adversarial examples with the noise addition-then-denoising method
 In this method, we utilyze the spatial instability of adversarial examples, i.e.,
 its label is likely to be changed after noise addition-then-denoising.
 Hui Zeng, 202106

 """

import matplotlib.pyplot as plt

import numpy as np
import os.path
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from FFDNet.models import FFDNet
from FFDNet.utils import normalize
from scipy.io import savemat,loadmat

import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def FFDNet_AddDe(imorig, sigma):
    """
    Process the input data with noise addition-then-denoising.
    imorig: input image, [0 255], RGB sequence, CxHxW

    """

    # CxHxW to 1xCxHxW
    imorig = np.expand_dims(imorig, 0)

    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
    if sh_im[2] % 2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
    if sh_im[3] % 2 == 1:
        expanded_w = True
        imorig = np.concatenate((imorig, imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

    imorig = normalize(imorig)
    # Add noise N(0, sigma^2)
    imnoisy = imorig + np.random.randn(*imorig.shape)*sigma
    imnoisy = torch.Tensor(imnoisy)

    # Denoising, test mode of FFDNet
    with torch.no_grad():
        imnoisy = Variable(imnoisy.type(dtype))
        nsigma = Variable(torch.FloatTensor([sigma]).type(dtype))

        # Estimate noise and subtract it to the input image
        im_noise_estim = modelFFD(imnoisy, nsigma)
        outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)

    if expanded_h:
        outim = outim[:, :, :-1, :]
    if expanded_w:
        outim = outim[:, :, :, :-1]

    return outim

if __name__ == "__main__":
    # set P_fa1 before run this code
    P_fa1 = 0.05

    # instantiate a resnet model for classification
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    # end of instantiate a resnet model

    # load a FFDNet model for denoising
    model_fn = 'FFDNet/net_rgb.pth'
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)
    # Create model
    net = FFDNet(num_input_channels=3)
    # Load saved weights
    state_dict = torch.load(model_fn)
    device_ids = [0]
    modelFFD = nn.DataParallel(net, device_ids=device_ids).cuda()
    modelFFD.load_state_dict(state_dict)
    # Sets the model in evaluation mode (e.g. it removes BN)
    modelFFD.eval()
    # Sets data type according to CPU or GPU modes
    dtype = torch.cuda.FloatTensor
    # end of instantiate a FFDnet model

    # From the experiment on a number of benign images, we have the approximate function:
    # P_fa1 = sigma / 70.7
    sigma = 70.7 * P_fa1

    imagepath = "images/ori/10.png"
    # imagepath = "images/PGD01/10.png"
    # imagepath = "images/PGD04/10.png"
    image = cv2.imread(imagepath)
    plt.imshow(image[:,:,::-1])
    plt.show()

    # important: the output of cv.imread() is BGR, HxWxC
    # whereas the model was trained with RGB, CxHxW
    image = image[:,:,::-1]
    image = np.transpose(image, (2, 0, 1))
    image_numpy = image

    image = np.asarray(image, dtype=np.float32)/255.
    image = ep.from_numpy(fmodel.dummy, image).raw              # type: ignore
    images= ep.astensors(image.unsqueeze(0))[0]

    # Top-1 label of the probe image
    res_ori = fmodel(images).argmax(axis=-1).numpy()
    res = res_ori[0]

    # noise addition-then denoising
    # image_numpy: [0 255]; sigma: [0 255]
    img_denoise = FFDNet_AddDe(image_numpy, sigma / 255)
    temp = ep.astensors(img_denoise)[0]

    # Top-1 label after noise addition-then denoising
    ress = fmodel(temp).argmax(axis=-1).numpy()
    res_denoise = ress[0]

    # If the label has been changed, then the probe image is adversarial,
    # Otherwise it is benign.
    if res == res_denoise:
        print("Benign image")
    else:
        print("Adversarial image")
