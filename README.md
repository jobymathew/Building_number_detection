# Building Number Detection

This project involves some image processing operations to extract digits from an image with a house number
and predicting those digits extracted.

Prediction is done using the K Nearest Neighbour Classifier.

## Dependencies

Python 3.0 or higher

OpenCV 3.10

## Implementation

1. Image Preprocessing:

 ```bash
 # Function to pre process the image
def preprocess(img):
# Converting to gray scale
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blurring the image
blur = cv2.blur(grayimg, (3,3))
# Getting the edges
edges = cv2.Canny(blur, 180, 255)
return edges
```

2. Finding the contours: The image will have a lot of noise contours along with our digits as we find the
contours of the preprocessed image. However, we only need the contours of the digits. They were removed by defining very specific criterias.

## Accuracy of the model

The accuracies obtained on cross validation are as shown:

![Cross Validation](https://github.com/jobymathew/Building_number_detection/blob/main/Validation_results/KnnAccuracy.png?raw=true)

## Results obtained on the Validation Data

Image 1

![Area1](https://github.com/jobymathew/Building_number_detection/blob/main/Validation_results/DetectedArea01.jpg?raw=True)

Prediction of the model: 43

Image 2

![Area1](https://github.com/jobymathew/Building_number_detection/blob/main/Validation_results/DetectedArea01.jpg?raw=True)

Prediction of the model: 35

Image 3

![Area1](https://github.com/jobymathew/Building_number_detection/blob/main/Validation_results/DetectedArea02.jpg?raw=True)

Prediction of the model: 94

Image 4

![Area1](https://github.com/jobymathew/Building_number_detection/blob/main/Validation_results/DetectedArea03.jpg?raw=True)

Prediction of the model: 302

Image 5

![Area1](https://github.com/jobymathew/Building_number_detection/blob/main/Validation_results/DetectedArea04.jpg?raw=True)

Prediction of the model: 70

Image 6

![Area1](https://github.com/jobymathew/Building_number_detection/blob/main/Validation_results/DetectedArea05.jpg?raw=True)

Prediction of the model: 26

## Accruacy of the model: 84.6%

## License

MIT License

Copyright (c) 2020 Joby Mathew

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
