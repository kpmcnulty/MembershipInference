import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dataset = np.load("./texas100.npz")
features= dataset['features']
labels = dataset['labels'] #onehot encoded
class Aggregator:
    def __init__(self, model_structure):
        self.model = model_structure

    def initialize_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def update_model(self, global_weights):
        self.model.set_weights(global_weights)


class Participant:
    def __init__(self, local_dataset):
        self.local_dataset = local_dataset
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_shape=(784,)))  # Adjust input shape as needed
        self.model.add(Dense(10, activation='softmax'))

    def train_local_model(self, num_epochs):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.local_dataset[0], self.local_dataset[1], epochs=num_epochs, batch_size=32)

    def get_weights(self):
        return self.model.get_weights()