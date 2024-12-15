# I

# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Load the data from the pickle file
# data_dict = pickle.load(open('./selfdata.pickle', 'rb'))

# # Filter out entries that do not have 84 landmarks
# filtered_data = [d for d in data_dict['data'] if len(d) == 84]
# filtered_labels = [data_dict['labels'][i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == 84]

# # Convert to numpy arrays
# data = np.array(filtered_data)
# labels = np.array(filtered_labels)

# # Print lengths to verify
# # print("Filtered data lengths:")
# # for i in data:
# #     print(len(i))

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Initialize and train the model
# model = RandomForestClassifier()
# model.fit(x_train, y_train)

# # Make predictions
# y_predict = model.predict(x_test)

# # Calculate the accuracy
# score = accuracy_score(y_predict, y_test)
# print('{}% of samples were classified correctly!'.format(score * 100))

# # Save the trained model to a file
# with open('selfmodel.p', 'wb') as f:
#     pickle.dump({'model': model}, f)

# II

# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Load the data from the pickle file
# data_dict = pickle.load(open('./selfdata.pickle', 'rb'))

# # Filter out entries that do not have 84 landmarks
# filtered_data = [d for d in data_dict['data'] if len(d) == 84]
# filtered_labels = [data_dict['labels'][i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == 84]

# # Convert to numpy arrays
# data = np.array(filtered_data)
# labels = np.array(filtered_labels)

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Define the model and hyperparameters to tune
# model = RandomForestClassifier()
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Use GridSearchCV to find the best hyperparameters
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(x_train, y_train)

# # Get the best model
# best_model = grid_search.best_estimator_

# # Make predictions
# y_predict = best_model.predict(x_test)

# # Calculate the accuracy
# score = accuracy_score(y_predict, y_test)
# print('{}% of samples were classified correctly!'.format(score * 100))

# # Save the trained model to a file
# with open('selfmodel.p', 'wb') as f:

#     pickle.dump({'model': best_model}, f)

# III 3 class neural net

# import pickle
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load the data from the pickle file
# data_dict = pickle.load(open('./selfdata.pickle', 'rb'))

# # Filter out entries that do not have 84 landmarks
# filtered_data = [d for d in data_dict['data'] if len(d) == 84]
# filtered_labels = [data_dict['labels'][i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == 84]

# # Ensure data is in numeric format and there are no string types
# filtered_data = np.array(filtered_data, dtype=np.float32)
# filtered_labels = np.array(filtered_labels, dtype=np.int32)

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels)

# # Define the neural network architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(84,)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(3, activation='softmax')  # Adjusted for 3 classes
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("Test Accuracy:", test_accuracy)

# # Save the trained model to a file
# model.save('selfmodel.h5')

#IV

# import pickle
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # Load the data from the pickle file
# data_dict = pickle.load(open('./data.pickle', 'rb'))

# # Filter out entries that do not have 84 landmarks
# filtered_data = [d for d in data_dict['data'] if len(d) == 84]
# filtered_labels = [data_dict['labels'][i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == 84]

# # Ensure data is in numeric format and there are no string types
# filtered_data = np.array(filtered_data, dtype=np.float32)

# # Encode the labels as integers
# label_encoder = LabelEncoder()
# filtered_labels = label_encoder.fit_transform(filtered_labels)

# # Check the number of classes
# num_classes = len(label_encoder.classes_)

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels)

# # Define the neural network architecture with regularization and dropout
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(84,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Use early stopping to prevent overfitting
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model
# model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("Test Accuracy:", test_accuracy)

# # Save the trained model to a file
# model.save('complete_model.h5')

#V
# import pickle
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # Load the data from the pickle file
# data_dict = pickle.load(open('./selfdata_augmented.pickle', 'rb'))

# # Filter out entries that do not have 84 landmarks (assuming each landmark has x and y coordinates, making 42 points)
# filtered_data = [d for d in data_dict['data'] if len(d) == 84]
# filtered_labels = [data_dict['labels'][i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == 84]

# # Ensure data is in numeric format and there are no string types
# filtered_data = np.array(filtered_data, dtype=np.float32)

# # Encode the labels as integers
# label_encoder = LabelEncoder()
# filtered_labels = label_encoder.fit_transform(filtered_labels)

# # Check the number of classes
# num_classes = len(label_encoder.classes_)

# # Reshape data to add a single channel dimension for Conv1D
# filtered_data = filtered_data.reshape(-1, 84, 1)

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels)

# # Define the neural network architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(84, 1)),
#     tf.keras.layers.MaxPooling1D(2),
#     tf.keras.layers.Conv1D(128, 3, activation='relu'),
#     tf.keras.layers.MaxPooling1D(2),
#     tf.keras.layers.Conv1D(256, 3, activation='relu'),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Use early stopping to prevent overfitting
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model
# model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("Test Accuracy:", test_accuracy)

# # Save the trained model to a file
# model.save('selfmodel_augmented.h5')

import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
data_dict = pickle.load(open('./selfdata_augmented.pickle', 'rb'))

# Filter out entries that do not have 168 values (84 landmarks * 2 coordinates)
filtered_data = [d for d in data_dict['data'] if len(d) == 168]
filtered_labels = [data_dict['labels'][i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == 168]

# Check if there's any data left after filtering
if len(filtered_data) == 0:
    raise ValueError("No data left after filtering. Check if your data has the correct number of landmarks.")

# Ensure data is in numeric format and there are no string types
filtered_data = np.array(filtered_data, dtype=np.float32)

# Encode the labels as integers
label_encoder = LabelEncoder()
filtered_labels = label_encoder.fit_transform(filtered_labels)

# Check the number of classes
num_classes = len(label_encoder.classes_)

# Reshape data to add a single channel dimension for Conv1D
filtered_data = filtered_data.reshape(-1, 84, 2)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels)

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(84, 2)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(256, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Use early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)

# Save the trained model to a file
model.save('selfmodel_augmented.h5')
