import numpy as np
import cv2
import os

# creating output folder
os.mkdir('output')
abspath = os.path.abspath("../../student")
"""
Training 
"""

# Number of train images
counts = []
k=0
digits_list = []
images = []
for i in range(10):
    count = 0
    for file in os.listdir(f'{abspath}/train/{i}/'):
        image = cv2.imread(f'{abspath}/train/{i}/{file}')
        # Resizing the image to 20,20
        image = cv2.resize(image,(20,20), interpolation=cv2.INTER_AREA)
        # Converting to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Getting the threshold
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Flattenin the image
        flatten = thresh.reshape((-1, 400)).astype(np.float32)
        digits_list.append(flatten)
        images.append(image)
        count += 1
    counts.append(count)

# Converting to numpy array
digits = np.array(digits_list)    

# Setting up the train daata    
X = digits.copy()
X = X.reshape((-1, 400))   

# Setting up labels 
k = np.arange(10)
y = np.repeat(k, counts)[:, np.newaxis]

# KNN Model
knn = cv2.ml.KNearest_create()
knn.train(X, cv2.ml.ROW_SAMPLE, y)

"""
Functions
"""

# Function to predict the given image
def predict_digit(image):
    # Resizing
    image = cv2.resize(image, (20,20), interpolation=cv2.INTER_CUBIC)
    # Getting the grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Getting the treshold
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Flattening the image
    flatten = thresh.reshape((-1, 400)).astype(np.float32)
    digit = np.array(flatten)
    # Finding the prediction
    ret, result, neighbours, dist = knn.findNearest(digit, k=5)
    return int(result[0][0])

# Function to check if two bounded images are similar
def is_similar(ordered, i):
    similar = False
    if i > 0:
        if ordered[i][0] > ordered[i-1][0] and ordered[i][0] < ordered[i-1][2]+ordered[i-1][0]:
            similar = True
    return similar

# Function to obtain the bounds of the detected area
def get_bounds(data):
    x = min([digit[0] for digit in data])    
    x2 = max([digit[0]+digit[2] for digit in data])   
    y = min([digit[1] for digit in data])
    y2 = max([digit[1]+digit[3] for digit in data])
    return x, y, x2, y2

# Function to return valid bounds by removing noises
def get_valid_bounds(hierarchies, digit_bounds):
    valid_bounds = []
    # Checking only if there are more than 2 bounds
    if len(hierarchies) > 2:
        # Selecting the ones with -1 if there are more than 3 of them
        if hierarchies.count(-1) >= 3:
            indices = [i for i, val in enumerate(hierarchies) if val == -1]
            for index in indices:
                valid_bounds.append(digit_bounds[index])
            # Selecting the remaning ones if they are not present already
            if (len(hierarchies)-hierarchies.count(-1)) > 1:
                indices = [i for i, val in enumerate(hierarchies) if val != -1]
                for index in indices:
                    if digit_bounds[index] not in valid_bounds:
                        valid_bounds.append(digit_bounds[index])
        # Sorting and choosing the ones which are in close hierarchy
        else:
            indices = sorted(hierarchies)
            first_index = hierarchies.index(indices[0])
            valid_bounds.append(digit_bounds[first_index])
            for i in range(len(indices)-1):
                if indices[i]+12 > indices[i+1]:
                    index = hierarchies.index(indices[i+1])
                    valid_bounds.append(digit_bounds[index])
                else:
                    break
    else:
        valid_bounds = digit_bounds
    return valid_bounds

# Function to pre process the image
def preprocess(img):
    # Converting to gray scale
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurring the image
    blur = cv2.blur(grayimg, (3,3))
    # Getting the edges 
    edges = cv2.Canny(blur, 180, 255)
    return edges

# Finding the bounds of the image using contours
def find_bounds(img, im, lower_limit, upper_limit, retr):
    digit_bounds, hierarchies = [], []
    _, cnts, hierarchy = cv2.findContours(im, retr, cv2.CHAIN_APPROX_SIMPLE)
    # Treshold for filtering light
    light = 20
    for i, cnt in enumerate(cnts):
        # Getting the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        # Getting the roi
        roi = im[y:y+h, x:x+w]
        roiDisplay = img[y:y+h, x:x+w]
        if (h < light or w < light):
            continue
        if(roi.shape[0] - roi.shape[1] >= lower_limit and roi.shape[0] - roi.shape[1] <= upper_limit) and cv2.contourArea(cnt) > 40:
            hierarchies.append(hierarchy[0][i][0])
            digit_bounds.append(cv2.boundingRect(cnt))

    return digit_bounds, img, hierarchies

"""
House Number detection and prediction
"""

for file in os.listdir(f"{abspath}/test/"):
    if file.endswith('.jpg') or file.endswith('.png'):
        name = file[4:6]
        img = cv2.imread(f'{abspath}/test/{file}')
        digit_bounds = []
        is_scaled = False
        # upper and lower limit for filtering unwanted contours
        upper_limit, lower_limit = 80, 10
        # Scaling the image and setting a new upper and lower limit
        if img.size < 160000:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            upper_limit = img.shape[0]
            lower_limit = img.shape[0]//17
            is_scaled = True

        # Getting the preprocessed image
        im = preprocess(img)

        # Getting the image bounds
        digit_bounds = find_bounds(img, im, lower_limit, upper_limit, cv2.RETR_EXTERNAL)

        # Repeating with retr tree if list is empty
        if digit_bounds[0] == []:
            digit_bounds = find_bounds(img, im, lower_limit, upper_limit, cv2.RETR_TREE)

        # Reducing the lower bound for images with less pixels if list is still empty
        if digit_bounds[0] == []:
            # Setting new upper and lower limit
            upper_limit = img.shape[0]
            lower_limit = 10

            im = preprocess(img)
            digit_bounds = find_bounds(img, im, lower_limit, upper_limit, cv2.RETR_EXTERNAL)

            # Repeating with retr tree if list is empty
            if digit_bounds[0] == []:
                digit_bounds = find_bounds(img, im, lower_limit, upper_limit, cv2.RETR_TREE)

        # Removing noise bounds
        valid_bounds = get_valid_bounds(digit_bounds[2], digit_bounds[0])

        # Getting the bouding box dimensions
        x, y, x2, y2 = get_bounds(valid_bounds)

        x-=3
        y-=3
        x2+=2
        y2+=3
        # Getting the rectangle with the numbers
        detected_area = img[y:y2, x:x2]
        w = x2-x
        h = y2-y
        # Dividing by two if the image was scaled
        if is_scaled:
            x //= 2
            y //=2
            w //=2
            h //=2  

        # Saving the bound dimensions
        # print(f'BoundingBox{name}: ({x},{y},{w},{h})')
        with open(f'output/BoundingBox{name}.txt', 'w') as bound_file:
            bound_file.write(f'({x},{y},{w},{h})')
            bound_file.close()

        # Preparing the detected area and marking the numbers
        im = preprocess(detected_area)
        digit_bounds = []
        _, cnts, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Treshold for filtering light
        light = 20

        for i, cnt in enumerate(cnts):
            # Getting the bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            # Getting the roi
            roi = im[y:y+h, x:x+w]
            roiDisplay = img[y:y+h, x:x+w]
            if (h < light or w < light):
                continue
            if(roi.shape[0] - roi.shape[1] >= 10 and roi.shape[0] - roi.shape[1] <= 200) and cv2.contourArea(cnt) > 40:
                digit_bounds.append(cv2.boundingRect(cnt))

        # Getting the x values
        x_values = [val[0] for val in digit_bounds]
        # Sorting the values
        x_values.sort()
        ordered = []
        # Sorting the digit image bounds
        for val in x_values:
            for bounds in digit_bounds:
                if bounds[0] == val:
                    ordered.append(bounds)

        predict_digits = []
        for i, val in enumerate(ordered):
            x, y, w, h = val
            # Ignoring similar bounds
            if not is_similar(ordered, i):
                cv2.rectangle(detected_area, (x, y), (x + w, y + h), (0, 255, 0), 2)
                x -= 3
                y -= 3
                w += 3
                h += 3
                roi = detected_area[y:y+h, x:x+w]
                # Getting the predicted output
                num = predict_digit(roi)
                predict_digits.append(num)
        # Saving the detected area and the building number
        cv2.imwrite(f'output/DetectedArea{name}.jpg', detected_area)
        with open(f'output/House{name}.txt', 'w') as house_file:
            house_file.write(f'Building {"".join(str(digit) for digit in predict_digits)}')
            house_file.close()
        # print('Building', ''.join(str(digit) for digit in predict_digits ))