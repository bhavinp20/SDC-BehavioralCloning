import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Cropping2D, Conv2D, Lambda

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
steering_angles = []

for line in lines:
    for i in range(3):
        img_source_path = line[i]
        img_file_name = img_source_path.split('\\')[-1]
        img_current_path = 'data/IMG/' + img_file_name
        img = cv2.imread(img_current_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        center_steering_angle = float(line[3])
        steering_angles.append(center_steering_angle)

X_train = np.array(images)
y_train = np.array(steering_angles)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

# Save the model
model.save('model.h5')

print('Done! Model Saved!')

# Print the model summary
model.summary()