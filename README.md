# Face Frontalization

This project generates frontal faces of any human being's faces.

![](./Git_images/Untitled-7b380c28-e8af-41b4-8475-04d7a9ec73af.png)

![](./Git_images/Untitled-bd2972f3-9acd-4fa3-af88-616cc0b02b5b.png)

![](./Git_images/Untitled-e29b4a05-cae9-4dde-b180-115b462ce453.png)

![](./Git_images/Untitled-dd10c67f-d76c-4152-a275-5607d588070c.png)

# Data

Data is consisted of 300 korean face data

**You can download data here**

[AI 오픈 이노베이션 허브](http://aihub.or.kr/)

![](./Git_images/Untitled-ffeeea0f-20d6-4997-a6ed-2259344f0213.png)

![](./Git_images/Untitled-fa9a5245-a71e-47dc-9e08-1daff4b429ac.png)

# Data Preprocessing

## Face Crop

Crop only face pixels of original image to **maximize the feature extraction efficiency**

Used functions and pretrained models in **Dlib** to crop faces

![](./Git_images/Untitled-ec76aa17-6d43-4511-b704-ec4ad904cd9c.png)

First, we try cropping image by using **Frontal face detector by Dlib**
```python
face_detector = dlib.get_frontal_face_detector()
rects = face_detector(image, 1)
```
If **Frontal face detector** does not find faces, try finding by using **MMOD human face detector**

```python
if len(rects) == 0:
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')
    rects = cnn_face_detector(image, 1)
```

Use 68 face landmark detections to get the entire face

```python
dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
face_regressor = dlib.shape_predictor(dlib_landmark_model)
pts = face_regressor(image, faceBoxRectangleS).parts()
pts = np.array([[pt.x, pt.y] for pt in pts]).T
roi_box = parse_roi_box_from_landmark(pts)
```
And crop image using roi box

```python
cropped_image = crop_img(image, roi_box)
```
## Face Crop Result

In Train datasets, **we did not use MMOD** CNN crop because it takes too much time (70seconds per 12 images).

So in total 1,700,000 face data, almost half is lost because in train datasets using Frontal Face detection. But still making the total amount of dataset 720,000 images.

The image below shows the Distribution of face tilt degree.

![](./Git_images/Untitled-3fa3a8d1-89c1-44dc-8552-74660a1bc0f9.png)

### Train Input

![](./Git_images/Untitled-f8b70783-a082-4b1c-b62d-981c040c7340.png)

### Train Label

![](./Git_images/Untitled-39a27425-8087-4006-9ff9-05443f18a911.png)

# DataGenerator

Since the total data is 720,000 images, memory gets full rising an error when the data is loaded by numpy. So by using **datagenerator.py**, model **loads images every batch** to use memory efficiently.

# Model

We tried different kinds of models:

- **Autoencoder**
- **CVAE**
- **VGG16 Face**
- **Unet**
- **DCGan**
- **Pixel to Pixel Gan**

**And the best model is Autoencoder with VGG16 Face as encoder and adding Unet.**

## **CVAE**

**Conditional Variational Autoencoder**

This is the first model we tried and it did not work well. 

At first, we used different data to train and also did not crop faces.

So the output was too blury which seems like model could not make extract face features.

![](./Git_images/Untitled-445557de-3260-417d-97bd-ddce46dcd750.png)

![](./Git_images/Untitled-b811d811-efbb-494d-8f59-03f5c780a088.png)

So after this model, we decided to **input cropped faces**, not the entire image

## DCGan

**Deep Convolutional Generative Adversarial Networks(DCGAN)**

DCGan did not work quite well

![](./Git_images/Untitled-0c9c88a2-1610-4634-8c53-4a5c4c21ba2e.png)

![](./Git_images/Untitled-53c19532-efc1-482f-bd7f-45844f2e8b86.png)

## Autoencoder + Unet ( Best Model )

After research, we decided to use **VGG16 Face** (pretrained model which extract face features from image) as our encoder.

And by researching more, we found **Patch Gan** which uses **Unet** to connect layers of encoder and decoders to **retrieve lost data.** 

![](./Git_images/Untitled-c18c1b92-e7dd-4d31-8fa2-3a40a1c689f9.png)

![](./Git_images/Untitled-2fd54661-b453-40cd-90d8-f9a8a124e9eb.png)

**Encoder:  VGG16 Face**

**Decoder: Convolutional Layers using Unet**

## Pixel to Pixel Gan

![](./Git_images/Untitled-1cd80787-f4cf-4c0e-9a72-313cc1725954.png)

![](./Git_images/Untitled-3e87669b-ad82-44e5-9d42-3268ec8ebb9a.png)


**Generator: Same as previous model (VGG16 + Unet)**

**Discriminator: Patch Gan**

# Dependency
- Python 3
- Python Image Library
- Anaconda

- Keras
- Tensorflow
- VGG16 Face
- Dlib 

- Django

- Numpy
- Opencv
- Matplotlib
- Glob
