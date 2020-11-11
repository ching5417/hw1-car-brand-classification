from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data.sampler import SubsetRandomSampler
import PIL.Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from RandAugment import RandAugment
import time
import argparse


parser = argparse.ArgumentParser()  # description='CNN'
parser.add_argument('--data_dir', type=str,
                    default='./cs-t0828-2020-hw1/training_data/training_data/',
                    help='path of training data')
parser.add_argument('--model_name', type=str, default='inception',
                    help='choose model[resnet,wide_resnet,inception]')
parser.add_argument('--num_classes', type=int, default=196,
                    help='number of classes in the dataset')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for training(default: 32)')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs to train for (default: 200)')
parser.add_argument('--feature_extract', type=bool, default=False,
                    help='flag for feature extracting(default: False)')
parser.add_argument('--input_size', type=int, default=299,
                    help='image input size (default: 299 (inception))')
parser.add_argument('--rand_m', type=int, default=9,
                    help='hyperparameter of RandAugment (default: 9)')
parser.add_argument('--rand_n', type=int, default=2,
                    help='hyperparameter of RandAugment (default: 2)')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate (default: 0.002)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight_decay (default: 5e-4)')


# Helper Functions
def train_model(model, dataloaders, criterion, optimizer,
                num_epochs, device, model_name, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                count += 1
                print(phase + "\t%d / %d" % (count, len(dataloaders[phase])),
                      end='\r')
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception
                    # because in training it has an auxiliary output.
                    # In train mode we calculate the loss
                    # by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/
                        # how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            print("")
            len_dataset = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / len_dataset
            epoch_acc = running_corrects.double() / len_dataset

            print('{} Loss: {:.4f} Acc: {:.4f}'
                  .format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        name_str = str(time.strftime('%Y%m%d_%H%M')) + '_' + model_name
        if epoch == 100:
            model_path = './model/' + name_str + '_100epochs.pt'
            torch.save(best_model_wts, model_path)
        if epoch == 150:
            model_path = './model/' + name_str + '_150epochs.pt'
            torch.save(best_model_wts, model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'
          .format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    time_str = str(time.strftime('%Y%m%d_%H%M')) + '_'
    model_path = './model/' + time_str + model_name + '.pt'
    torch.save(best_model_wts, model_path)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


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


# Load Data
class MyDataSet(Dataset):
    def __init__(self, main_dir, df, transform):
        self.main_dir = main_dir
        self.df = df
        self.transform = transform
        imgs_dir = main_dir  # + 'training_data/training_data/'
        self.total_imgs = ["0"*(6-len(str(fileid)))+str(fileid)+".jpg"
                           for fileid in df.id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])

        image = PIL.Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        img_id = int(self.total_imgs[idx].split('.')[0])
        label_tmp = self.df['label_n'][self.df[self.df.id == img_id].index].values[0]
        label = torch.from_numpy(np.array(label_tmp))
        return tensor_image, label


def data_preprocess(data_dir, rand_m, rand_n, batch_size):
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            RandAugment(rand_m, rand_n),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
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
        class_dict[str(i)] = c
    df_class = pd.DataFrame.from_dict(class_dict, orient="index")
    df_class.to_csv("./class_dict.csv")

    # splict data to training set and validation set
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label_n)

    # prepare training dataset and val dataset
    train_dataset = MyDataSet(data_dir, train_df,
                              transform=data_transforms['train'])
    val_dataset = MyDataSet(data_dir, val_df, transform=data_transforms['val'])

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               drop_last=False, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    drop_last=False)

    # Dataloader Dictionary
    dataloaders_dict = {'train': train_loader, 'val': validation_loader}

    return dataloaders_dict


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    args = parser.parse_args()

    # Initialize the model for this run
    model_ft, input_size = initialize_model(args.model_name, args.num_classes,
                                            args.feature_extract,
                                            use_pretrained=True)
    # Print the model we just instantiated
    print(model_ft)

    # data preprocessing
    dataloaders_dict = data_preprocess(args.data_dir,
                                       args.rand_m,
                                       args.rand_n,
                                       args.batch_size)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad is True:
                print("\t", name)

    # create optimizer, observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=args.learning_rate,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
                                 optimizer_ft, args.num_epochs,
                                 device, args.model_name,
                                 is_inception=(args.model_name == "inception"))
