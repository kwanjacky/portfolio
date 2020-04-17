import argparse
import json
import matplotlib.pyplot as plt

import torch

import PIL
from PIL import Image
import numpy as np


def load_checkpoint(filepath):
  '''
  Loads a checkpoint and rebuilds the model
  '''
  checkpoint = torch.load(filepath, map_location='cpu')
  model = checkpoint['model']
  model.classifier = checkpoint['classifier']
  model.load_state_dict(checkpoint['state_dict'], strict=False)
  model.class_to_idx = checkpoint['class_to_idx']
  optimizer = checkpoint['optimizer']
  epochs = checkpoint['epochs']
  
  for param in model.parameters():
      param.requires_grad = False
      
  return model, checkpoint['class_to_idx']


def process_image(image):
  ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
      returns an Numpy array
  '''
  # Process a PIL image for use in a PyTorch model
  out_img = Image.open(image)
  
  # resize image to 256 pixels and keep aspect ratio
  new_size = (256, 256)
  out_img.thumbnail(new_size)
      
  # crop out center 224 x 224 from 256 x 256
  orig_side = 256
  crop_side = 224
  left = (orig_side - crop_side)/2
  top = (orig_side - crop_side)/2
  right = (orig_side + crop_side)/2
  bottom = (orig_side + crop_side)/2
  out_img=out_img.crop((left, top, right, bottom))
  
  # convert color channels to floats 0-1
  out_array = np.array(out_img) / 255
  
  # normalize one color channel at a time
  means = np.array([0.485, 0.456, 0.406])
  stdevs = np.array([0.229, 0.224, 0.225])
  out_array = (out_array - means) / stdevs
      
  # change color channel (i.e. transpose) from PIL to torch format
  out_array = out_array.transpose((2, 0, 1))
  
  return out_array

def imshow(image, ax=None, title=None):
  """
  Imshow for Tensor
  """
  if ax is None:
      fig, ax = plt.subplots()
  
  # PyTorch tensors assume the color channel is the first dimension
  # but matplotlib assumes is the third dimension
  image = image.numpy().transpose((1, 2, 0))
  
  # Undo preprocessing
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  image = std * image + mean

  # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
  image = np.clip(image, 0, 1)
      
  ax.imshow(image)
  
  return ax    

def predict(image_path, model, topk=5):
  ''' 
  Predict the class (or classes) of an image using a trained deep learning model.
  '''
  
  # TODO: Implement the code to predict the class from an image file
  processed_im = process_image(image_path)

  im = torch.FloatTensor(processed_im)
  im = im.unsqueeze(0)

  ps = torch.exp(model(im))
  top_p, top_class = ps.topk(topk, dim=1)
  
  return top_p.numpy()[0], top_class.numpy()[0]

def main(args):
  '''
  Predict using pretrain model

  Parameters
  -----------
  workdir :       Working directory where checkpoint fname will be saved

  checkpoint_fname : Checkpoint outfile fname to use

  predict_dir :   Root directory where image to predict is stored

  predict_fname : Fname of image to predict flower species

  label_map :     Json file where number-to-species label mapping is stored

  topk :     	  Top k likely flower species to return

  '''

  ## Loading the checkpoint
  print("\nLoading checkpoint...")

  workdir = args.workdir
  checkpoint_fname = args.checkpoint_fname
  predict_dir = args.predict_dir
  predict_fname = args.predict_fname
  label_map = args.label_map
  topk = args.topk

  ### Load label mapping
  with open(label_map, 'r') as f:
      cat_to_name = json.load(f)

  model, class_to_idx = load_checkpoint(workdir+checkpoint_fname)

  ## Class Prediction
  image_path = '%s%s' %(predict_dir, predict_fname)
  im = process_image(image_path)

  probs, classes = predict(image_path, model, topk)

  # show ground truth vs. top predictions
  ground_truth_idx = predict_dir.split('/')[-1]
  print("\nGround Truth: %s" %cat_to_name[ground_truth_idx].title())
  print("\nTop %s predictions..." %topk)
  for i in range(len(classes)):
  	for k, v in class_to_idx.items():
  		if str(v)==str(classes[i]):
  			print(f'{probs[i]:.3f}', k, cat_to_name[str(k)].title())

if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument('workdir', type=str, 
                      default = '/home/workspace/aipnd-project/')  
  parser.add_argument('checkpoint_fname', type=str, 
                      default = 'checkpoint_v0.pth')  
  parser.add_argument('predict_dir', type=str, 
                      default = '/home/workspace/aipnd-project/flowers/test/10/', 
                      help = 'Root directory where image to predict is stored')
  parser.add_argument('predict_fname', type=str, 
                      default = 'image_07090.jpg', help = 'Default: Globe Thistle')
  parser.add_argument('topk', type=int, 
                      default = 5)
  parser.add_argument('label_map', type=str, 
                      default = 'cat_to_name.json')  

  args = parser.parse_args()
  main(args)
