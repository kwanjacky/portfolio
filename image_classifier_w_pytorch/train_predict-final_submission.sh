# train transfer learning net
# Based on Image Classifier Project_6-23b notebook
cd ImageClassifier

workdir=/home/workspace/ImageClassifier/
data_dir=flowers
learning_rate=0.001
hidden_units=500
epochs=10
batch_size=64
label_map=cat_to_name.json
version=v2
checkpoint_fname=checkpoint.pth

time python train.py ${workdir} ${data_dir} ${learning_rate} ${hidden_units} ${epochs} ${batch_size} ${label_map} ${version} ${checkpoint_fname}

# Training begins...
# Epoch 1/10.. Train loss: 3.359.. Validation loss: 1.643.. Validation accuracy: 0.575
# Epoch 1/10.. Train loss: 1.922.. Validation loss: 0.914.. Validation accuracy: 0.762
# Epoch 2/10.. Train loss: 1.399.. Validation loss: 0.612.. Validation accuracy: 0.836
# Epoch 2/10.. Train loss: 1.046.. Validation loss: 0.518.. Validation accuracy: 0.860
# Epoch 2/10.. Train loss: 1.047.. Validation loss: 0.445.. Validation accuracy: 0.879
# Epoch 3/10.. Train loss: 0.895.. Validation loss: 0.419.. Validation accuracy: 0.891
# Epoch 3/10.. Train loss: 0.892.. Validation loss: 0.381.. Validation accuracy: 0.888
# Epoch 4/10.. Train loss: 0.795.. Validation loss: 0.381.. Validation accuracy: 0.902
# Epoch 4/10.. Train loss: 0.774.. Validation loss: 0.365.. Validation accuracy: 0.900
# Epoch 4/10.. Train loss: 0.770.. Validation loss: 0.370.. Validation accuracy: 0.892
# Epoch 5/10.. Train loss: 0.701.. Validation loss: 0.350.. Validation accuracy: 0.905
# Epoch 5/10.. Train loss: 0.738.. Validation loss: 0.333.. Validation accuracy: 0.906
# Epoch 6/10.. Train loss: 0.691.. Validation loss: 0.397.. Validation accuracy: 0.902
# Epoch 6/10.. Train loss: 0.725.. Validation loss: 0.335.. Validation accuracy: 0.916
# Epoch 6/10.. Train loss: 0.674.. Validation loss: 0.309.. Validation accuracy: 0.917
# Epoch 7/10.. Train loss: 0.675.. Validation loss: 0.355.. Validation accuracy: 0.901
# Epoch 7/10.. Train loss: 0.574.. Validation loss: 0.378.. Validation accuracy: 0.906
# Epoch 7/10.. Train loss: 0.630.. Validation loss: 0.327.. Validation accuracy: 0.911
# Epoch 8/10.. Train loss: 0.643.. Validation loss: 0.316.. Validation accuracy: 0.911
# Epoch 8/10.. Train loss: 0.632.. Validation loss: 0.326.. Validation accuracy: 0.917
# Epoch 9/10.. Train loss: 0.655.. Validation loss: 0.351.. Validation accuracy: 0.919
# Epoch 9/10.. Train loss: 0.585.. Validation loss: 0.384.. Validation accuracy: 0.915
# Epoch 9/10.. Train loss: 0.618.. Validation loss: 0.316.. Validation accuracy: 0.925
# Epoch 10/10.. Train loss: 0.536.. Validation loss: 0.361.. Validation accuracy: 0.915
# Epoch 10/10.. Train loss: 0.555.. Validation loss: 0.371.. Validation accuracy: 0.916

# Running validation on the test set...
# Test loss: 0.364.. Test accuracy: 0.910

# Saving checkpoint to checkpoint_v2.pth...

# Training complete!

# real      24m21.950s
# user      22m38.190s
# sys       3m57.151s

# predict one image
workdir=/home/workspace/ImageClassifier/
checkpoint_fname=checkpoint_v2.pth
predict_dir=/home/workspace/ImageClassifier/flowers/test/10/
predict_fname=image_07090.jpg
topk=5
label_map=cat_to_name.json

time python predict.py ${workdir} ${checkpoint_fname} ${predict_dir} ${predict_fname} ${topk} ${label_map}

# Ground Truth: Globe Thistle

# Top 5 predictions...
# 1.000 10 Globe Thistle
# 0.000 29 Artichoke
# 0.000 14 Spear Thistle
# 0.000 38 Great Masterwort
# 0.000 92 Bee Balm

# real      0m1.849s
# user      0m1.383s
# sys       0m0.853s
