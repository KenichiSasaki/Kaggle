# CloudFiltering
## Feature Learning with CNN

- `Preprocess.py`: Script to generate dataset, dataset is a set of cropped images with 5 types labels (Fish, Flower, Gravel, Sugar, Non)
  - Saves the output with `data{0}_{1}.npy` ,`label{0}_{1}.npy`, {0}, {1} corresponds to number of labels and number of original images.
  - Uploaded the sample generated with 100 images (This even has 67 MB)
- `VGG7.py`: Script with 7 layers CNN (4 convolutional layers and 3 full connected layers)
  - Save the learned model at the end
- `VGG16.py`: Script Script with 16 layers CNN (13 convolutional layers and 3 full connected layers)
  - computationally intensive model than VGG7
  - Save the learned model at the end