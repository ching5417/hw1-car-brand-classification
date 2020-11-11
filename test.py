from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import PIL.Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import argparse

parser = argparse.ArgumentParser()  # description='CNN'
parser.add_argument('--data_dir', type=str,
                    default='./cs-t0828-2020-hw1/testing_data/testing_data/',
                    help='path of testing data')
parser.add_argument('--PATH', type=str,
                    default='./model/20201102_0700_inception.pt',
                    help='path of model_path')
parser.add_argument('--model_name', type=str, default='inception',
                    help='choose model[resnet,wide_resnet,inception]')
parser.add_argument('--num_classes', type=int, default=196,
                    help='number of classes in the dataset')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for testing(default: 32)')
parser.add_argument('--feature_extract', type=bool, default=False,
                    help='flag for feature extracting(default: False)')
parser.add_argument('--input_size', type=int, default=299,
                    help='image input size (default: 299 (inception))')


# Set Model Parameters’ .requires_grad attribute
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Initialize and Reshape the Networks
def initialize_model(model_name, num_classes,
                     feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement.
    # Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "wide_resnet_50":
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2',
                                  pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def test_model(model, dataloader, class_dict, is_inception=False):
    all_img_id = None
    prediction = None

    since = time.time()

    # Set model to evaluate mode
    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    count = 0
    for img_id, inputs in dataloader:
        count += 1
        print("test" + "\t%d / %d" % (count, len(dataloader)), end='\r')
        img_id = img_id.to(device)
        inputs = inputs.to(device)

        # zero the parameter gradients
        # optimizer.zero_grad()

        # forward
        # track history if only in train
        if is_inception:
            with torch.no_grad():
                # From https://discuss.pytorch.org/t/
                # how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                outputs = model(inputs)
                # outputs, aux_outputs  = model(inputs)
        else:
            with torch.no_grad():
                outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        if all_img_id is None:
            img_id_tmp = np.reshape(img_id.cpu().numpy(), [-1])
            preds_tmp = np.reshape(preds.cpu().numpy(), [-1])
            all_img_id = ["0"*(6-len(str(l)))+str(l)
                          for l in list(img_id_tmp)]
            prediction = [class_dict[l]
                          for l in list(preds_tmp)]
        else:
            img_id_tmp = np.reshape(img_id.cpu().numpy(), [-1])
            preds_tmp = np.reshape(preds.cpu().numpy(), [-1])
            all_img_id += ["0"*(6-len(str(l)))+str(l)
                           for l in list(img_id_tmp)]
            prediction += [class_dict[l]
                           for l in list(preds_tmp)]

    print()
    csv_out_name = "./output/output.csv"
    pd_tmp = pd.DataFrame({"id": all_img_id, "label": prediction})
    pd_tmp.to_csv(csv_out_name, index=False)
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'
          .format(time_elapsed // 60, time_elapsed % 60))

    return all_img_id, prediction


class TestDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        imgs_dir = main_dir
        self.total_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = PIL.Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        img_id_tmp = np.array(int(self.total_imgs[idx].split('.')[0]))
        img_id = torch.from_numpy(img_id_tmp)
        return img_id, tensor_image


# data preprocessing
def data_preprocess(data_dir, batch_size):
    # Just normalization for validation
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # load training id and labels table
    df = pd.read_csv('./cs-t0828-2020-hw1/training_labels.csv')
    # Encode label from text label to number label
    label_encoder = LabelEncoder()
    df['label_n'] = label_encoder.fit_transform(df['label'])
    class_dict = {}
    for i, c in enumerate(label_encoder.classes_):
        class_dict[i] = c

    # prepare testing dataset
    test_dataset = TestDataSet(data_dir, transform=data_transforms['test'])

    # Dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              drop_last=False)

    return class_dict, test_loader


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    args = parser.parse_args()

    # Initialize the model for this run
    model_ft, input_size = initialize_model(args.model_name, args.num_classes,
                                            args.feature_extract,
                                            use_pretrained=True)

    model_ft.load_state_dict(torch.load(args.PATH))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    class_dict, test_loader = data_preprocess(args.data_dir, args.batch_size)

    model_ft = model_ft.to(device)

    # test
    inception_check = (args.model_name == "inception")
    img_id, prediction = test_model(model_ft, test_loader, class_dict,
                                    is_inception=inception_check)
