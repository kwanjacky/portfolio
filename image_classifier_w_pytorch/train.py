import argparse
import json

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F

def main(args):
  '''
  Trains Transfer Learning vgg model using images in data_dir

  Based on Image Classifier Project_6-23b notebook

  Parameters
  -----------
  workdir :       Working directory where checkpoint fname will be saved

  data_dir :      Directory where images are stored in /train, /valid, and 
                  /test directories 

  learning_rate : Learning rate to use in training 

  hidden_units :  Number of nodes in hidden layer
  
  epochs :        Number of epochs to train on

  batch_size :    Batch size for number of images to train at a time

  label_map :     Json file where number-to-species label mapping is stored

  version :       Version number to use in checkpoint outfile fname

  checkpoint_fname : Checkpoint outfile fname to use

  '''

  ### Load the data
  print("\nLoading data...")
  workdir = args.workdir
  data_dir = args.data_dir
  learning_rate = args.learning_rate
  hidden_units = args.hidden_units
  epochs = args.epochs
  batch_size = args.batch_size
  label_map = args.label_map
  version = args.version
  checkpoint_fname = args.checkpoint_fname.replace('.pth','_'+version+'.pth')

  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'

  # Define transforms for the training, validation, and testing sets
  train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                        transforms.RandomResizedCrop(224), 
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

  valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

  test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

  # Load the datasets with ImageFolder
  train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
  valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
  test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

  # Using the image datasets and the trainforms, define the dataloaders
  trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
  validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

  ### Label mapping
  with open(label_map, 'r') as f:
      cat_to_name = json.load(f)

  ### Building and training the classifier

  # Use GPU if it's available
  if torch.cuda.is_available(): 
    device = torch.device("cuda")
    print("\nSet device to GPU")
  else: 
    device = torch.device("cpu")
    print("\nSet device to CPU")

  # train VGG model

  model = models.vgg11(pretrained=True)

  # Freeze parameters so we don't backprop through them
  for param in model.parameters():
      param.requires_grad = False
      
  # for densenet model 
  model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))

  criterion = nn.NLLLoss()

  # Only train the classifier parameters, feature parameters are frozen
  print("\nTraining begins...")
  optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

  model.to(device);

  steps = 0
  running_loss = 0
  print_every = 40
  for epoch in range(epochs):
      for inputs, labels in trainloader:
          steps += 1
          # Move input and label tensors to the default device
          inputs, labels = inputs.to(device), labels.to(device)
          
          optimizer.zero_grad()
          
          logps = model.forward(inputs)
          loss = criterion(logps, labels)
          loss.backward()
          optimizer.step()
          
          running_loss += loss.item()
          
          if steps % print_every == 0:
              valid_loss = 0
              accuracy = 0
              model.eval()
              with torch.no_grad():
                  for inputs, labels in validloader:
                      inputs, labels = inputs.to(device), labels.to(device)
                      logps = model.forward(inputs)
                      batch_loss = criterion(logps, labels)
                      
                      valid_loss += batch_loss.item()
                      
                      # Calculate accuracy
                      ps = torch.exp(logps)
                      top_p, top_class = ps.topk(1, dim=1)
                      equals = top_class == labels.view(*top_class.shape)
                      accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                      
              print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
              
              running_loss = 0
              model.train()


  # ## Testing network
  print('\nRunning validation on the test set...')

  test_loss = 0
  accuracy = 0
  model.eval()
  with torch.no_grad():
      for inputs, labels in testloader:
          inputs, labels = inputs.to(device), labels.to(device)
          logps = model.forward(inputs)
          batch_loss = criterion(logps, labels)
          
          test_loss += batch_loss.item()
          
          # Calculate accuracy
          ps = torch.exp(logps)
          top_p, top_class = ps.topk(1, dim=1)
          equals = top_class == labels.view(*top_class.shape)
          accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
          
  print(f"Test loss: {test_loss/len(testloader):.3f}.. "
        f"Test accuracy: {accuracy/len(testloader):.3f}")


  # ## Save the checkpoint
  print("\nSaving checkpoint to %s..." %checkpoint_fname)

  model.class_to_idx = train_data.class_to_idx

  # workdir = os.getcwd() #'/home/workspace/aipnd-project/'
  fname = checkpoint_fname
  checkpoint = {'input_size': 25088,
                'output_size': 102,
                'epochs': epochs,
                'batch_size': batch_size,
                'model': models.vgg11(pretrained=True),
                'classifier': model.classifier,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx
               }
     
  torch.save(checkpoint, workdir+fname)
  print("\nTraining complete!")


if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument('workdir', type=str, 
                      default = '/home/workspace/aipnd-project/')
  parser.add_argument('data_dir', type=str, 
                      default = 'flowers', 
                      help = 'Root directory where images are stored')
  parser.add_argument('learning_rate', type=float, 
                      default = 0.001)
  parser.add_argument('hidden_units', type=int, 
                      default = 500)               
  parser.add_argument('epochs', type=int, 
                      default = 20)
  parser.add_argument('batch_size', type=int, 
                      default = 64)
  parser.add_argument('label_map', type=str, 
                      default = 'cat_to_name.json')
  parser.add_argument('version', type=str, 
                      default = 'v0')
  parser.add_argument('checkpoint_fname', type=str, 
                      default = 'checkpoint.pth') 

  args = parser.parse_args()
  main(args)
