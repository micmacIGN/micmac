import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os
import cv2
import sys
import keras
import argparse
import numpy as np
import segmentation_models as sm
import albumentations as A
import imageio


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image) #, mask=mask)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)



def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)



parser = argparse.ArgumentParser(
    description="Computes sky mask on 480 x 360 px images")

parser.add_argument("-i", "--input", help="Input folder", default="./", type=str)
parser.add_argument("-o", "--output", help="Output folder", default="./", type=str)
parser.add_argument("-p", "--proba", help="Probabilistic mode", default=0, type=int)
parser.add_argument("-r", "--inv", help="Inverse prediction", default=0, type=int)
parser.add_argument("-t", "--thresh", help="Decision threshold", default=0.5, type=float)
parser.add_argument("-f", "--filter", help="Filtering", default=0.0, type=float)

args = parser.parse_args()

print("------------------------------------------")
print("Loading neural network model...")
print("------------------------------------------")

BACKBONE = 'efficientnetb3'
n_classes = 1
activation = 'sigmoid'
CLASSES = ['sky']

DATA_DIR = args.input
output_path = args.output
prob = args.proba
invert = args.inv
threshold = args.thresh
filtrage = args.filter

#DATA_DIR = './input/'
#x_test_dir = os.path.join(DATA_DIR, 'yann_test')
#output_path = './output'

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model.load_weights('best_model.h5')
preprocess_input = sm.get_preprocessing(BACKBONE)

test_dataset = Dataset(
    DATA_DIR, 
    classes=CLASSES, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

crop = A.CenterCrop(360, 480)

BIN = 10
HIST = [0]*256*BIN
OUTPUT_PATHS = []

print("------------------------------------------")
print("Prediction...")
print("------------------------------------------")

for i in range(len(test_dataset)):
    image = test_dataset[i]
    name = test_dataset.ids[i]
    image = np.expand_dims(image, axis=0)
    p = model.predict(image)

    unpadded = (crop(image=p.squeeze())['image'])*256
	
    for j in range(unpadded.shape[0]):
        for k in range(unpadded.shape[1]): 
            id = (int)(unpadded[j,k]*BIN)
            HIST[id] = HIST[id]+1

    name_out = f'{output_path}/{name}'
    OUTPUT_PATHS.append(name_out)
    print('Mask for', name, 'done ['+(str)(i+1)+'/'+(str)(len(test_dataset))+']')
    np.savetxt(name_out+".dat", unpadded)
	

F = [0]*len(HIST)
    
for h in range(1,len(F)):
	F[h] = F[h-1] + HIST[h-1]
	
for h in range(len(F)):
    F[h] = F[h]/F[len(F)-1]*255	
	
print("------------------------------------------")
print("Renormalizing and filtering images...")
print("------------------------------------------")
if (not prob) and (threshold < 0):
    threshold = F[(int)(0.5*255)]
    print("Automatic threshold: "+(str)(threshold))
else:
    threshold *= 255
	
filtre = (int)(((360+480)/2*filtrage)**2) 
if filtrage and (filtre > 0):
    print("Filtering threshold: "+(str)(filtre))
	

for i in range(len(OUTPUT_PATHS)):
	
    print(OUTPUT_PATHS[i])
	
    img = np.loadtxt(OUTPUT_PATHS[i]+".dat")
	
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            img[j,k] = F[(int)(img[j,k]*BIN)]
            if not prob:
                img[j,k] = (img[j,k] >= threshold)*255
                if (j==0) or (k==0) or (j==img.shape[0]) or (k==img.shape[1]):
                    img[j,k] = 0
            img[j,k] = 255 - img[j,k]

    if (filtre > 0) and (not prob):
        ret, labels = cv2.connectedComponents(np.uint8(img))
        for k in range(1, np.max(labels)+1):
            K = np.where(labels == k)
            if len(K[0]) < filtre:
                img[K] = 0
			
    if not invert:
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                img[j,k] = 255 - img[j,k]
				
    imageio.imwrite(OUTPUT_PATHS[i], np.uint8(img))

sys.exit()