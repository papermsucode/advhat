## Demo launch

ArcFace@ms1m-refine-v2 transformed to TensorFlow is available [here](https://drive.google.com/file/d/1fb70KgMRSmaEUF5cJ67BCD_DmTPCR5uJ/view?usp=sharing).

The command for demo launch:

`python3 demo.py PATH_TO_THE_DOWNLOADED_MODEL PATH_TO_THE_DIRECTORY_WITH_CLASS_CENTROIDS`

Centroids for the first 1000 classes of CASIA are in the "1000_from_CASIA" directory.

## Preparation of your own centroids

### Alignment

The dataset of your images has to be arranged in the following way:  

├── Person 1  
│   ├── Person_1_image_1.png  
│   ├── Person_1_image_2.png  
│   ├── Person_1_image_3.png   
│   └── Person_1_image_4.png  
├── Person 2  
│   ├── Person_2_image_1.png  
│   ├── Person_2_image_2.png  
│   ├── Person_2_image_3.png   
│   ├── Person_2_image_4.png  
│   └── Person_2_image_5.png  
├── Person 3  
│   ├── Person_3_image_1.png  
│   ├── Person_3_image_2.png  
...  

The command for images alignment:

`python3 alignment.py PATH_TO_DIRECTIRY_WITH_IMAGES PATH_FOR_THE_ALIGNED_IMAGES`

### Centroids calculation

Using directory with aligned images from the previous step, you can obtain centroids with the next command:

`python3 dumping.py PATH_TO_DIRECTORY_WITH_ALIGNED_IMAGES PATH_FOR_THE_CENTROIDS PATH_TO_THE_DOWNLOADED_MODEL`
