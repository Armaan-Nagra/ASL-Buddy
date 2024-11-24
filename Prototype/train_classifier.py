import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np  

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Load the data
# data_dict = pickle.load(open('./data.pickle', 'rb'))

# # Inspect the structure of `data_dict['data']`
# print("Type of data_dict['data']:", type(data_dict['data']))
# print("Number of samples:", len(data_dict['data']))
# print("First 5 samples:", data_dict['data'][:5])  # Print the first 5 elements for debugging

# # Ensure all samples in `data` are of consistent shape
# try:
#     data = np.asarray(data_dict['data'])
# except ValueError as e:
#     print("Error converting to NumPy array:", e)
#     # Fix inconsistent shapes
#     data = [np.array(sample) for sample in data_dict['data']]
#     max_length = max(len(sample) for sample in data)  # Find the longest sample
#     print("Normalizing data to length:", max_length)
#     data = np.array([np.pad(sample, (0, max_length - len(sample)), mode='constant') for sample in data])

# # Convert labels to a NumPy array
# labels = np.asarray(data_dict['labels'])

# # Split the data
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Train the Random Forest model
# model = RandomForestClassifier()
# model.fit(x_train, y_train)

# # Predict and calculate accuracy
# y_predict = model.predict(x_test)
# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly!'.format(score * 100))

# # Save the trained model
# with open('model.p', 'wb') as f:
#     pickle.dump({'model': model}, f)