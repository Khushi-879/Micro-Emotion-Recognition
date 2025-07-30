from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

TRAIN_DIR = r"C:\Users\HP\Desktop\train"
TEST_DIR = r"C:\Users\HP\Desktop\validation"

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels


train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)
print(train)

angry completed
disgust completed
fear completed
happy completed
neutral completed
sad completed
surprise completed
                                             image     label
0            C:\Users\HP\Desktop\train\angry\0.jpg     angry
1            C:\Users\HP\Desktop\train\angry\1.jpg     angry
2           C:\Users\HP\Desktop\train\angry\10.jpg     angry
3        C:\Users\HP\Desktop\train\angry\10002.jpg     angry
4        C:\Users\HP\Desktop\train\angry\10016.jpg     angry
...                                            ...       ...
28816  C:\Users\HP\Desktop\train\surprise\9969.jpg  surprise
28817  C:\Users\HP\Desktop\train\surprise\9985.jpg  surprise
28818  C:\Users\HP\Desktop\train\surprise\9990.jpg  surprise
28819  C:\Users\HP\Desktop\train\surprise\9992.jpg  surprise
28820  C:\Users\HP\Desktop\train\surprise\9996.jpg  surprise

[28821 rows x 2 columns]

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)
print(test)


ngry completed
disgust completed
fear completed
happy completed
neutral completed
sad completed
surprise completed
                                                 image     label
0       C:\Users\HP\Desktop\validation\angry\10052.jpg     angry
1       C:\Users\HP\Desktop\validation\angry\10065.jpg     angry
2       C:\Users\HP\Desktop\validation\angry\10079.jpg     angry
3       C:\Users\HP\Desktop\validation\angry\10095.jpg     angry
4       C:\Users\HP\Desktop\validation\angry\10121.jpg     angry
...                                                ...       ...
7061  C:\Users\HP\Desktop\validation\surprise\9806.jpg  surprise
7062  C:\Users\HP\Desktop\validation\surprise\9830.jpg  surprise
7063  C:\Users\HP\Desktop\validation\surprise\9853.jpg  surprise
7064  C:\Users\HP\Desktop\validation\surprise\9878.jpg  surprise
7065   C:\Users\HP\Desktop\validation\surprise\993.jpg  surprise

[7066 rows x 2 columns]

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
    return features

train_features = extract_features(train['image'])

0%|                                                                                        | 0/28821 [00:00<?, ?it/s]C:\Users\HP\AppData\Roaming\Python\Python310\site-packages\keras_preprocessing\image\utils.py:107: UserWarning: grayscale is deprecated. Please use color_mode = "grayscale"
  warnings.warn('grayscale is deprecated. Please use '
100%|███████████████████████████████████████████████████████████████████████████| 28821/28821 [00:43<00:00, 665.99it/s]

test_features = extract_features(test['image'])

100%|████████████████████████████████████████████████████████████████████████████| 7066/7066 [00:05<00:00, 1214.94it/s]

le = LabelEncoder()
le.fit(train['label'])


LabelEncoder?i
LabelEncoder()

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

model = Sequential()
model.add(Input(shape=(48, 48, 1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save model architecture as JSON
model_json = model.to_json()
with open("emotiondetector.json", 'w') as json_file:
    json_file.write(model_json)

# Save the entire model in the recommended Keras format
model.save("emotiondetector.keras")


from keras.models import load_model

# Load the trained model
model = load_model("emotiondetector.keras")  # Use the correct file name


label = ['angry','disgust','fear','happy','neutral','sad','surprise']

def ef(image):
    img = load_img(image,grayscale =  True )
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

import matplotlib.pyplot as plt
%matplotlib inline

image = r"C:\Users\HP\Desktop\detection\train\fear\5.jpg"
print("original image is of fear")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


