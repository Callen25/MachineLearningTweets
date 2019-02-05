from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
import pickle
import numpy

NAME = "Trump_tweeets-Final-Model_Testing8"

tensor_board = TensorBoard(log_dir='logs/{}'.format(NAME))

print('Loading data')

x = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

sequence_length = x.shape[1]
#Words in vocab + 1 for unrecognized words {0}
vocabulary_size = 6072
num_filters = 100

epochs = 100
batch_size = 60


X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=3)

inputs = Input(shape=(sequence_length,), dtype='int32')
#This is not really an embedding layer, since it's initalized to identity it just gives the one hot representation
embedding = Embedding(input_dim=vocabulary_size, output_dim=vocabulary_size, input_length=sequence_length, embeddings_initializer='identity', trainable=False)(inputs)

l_cov1 = Conv1D(num_filters, 2, activation='relu')(embedding)
l_cov2 = Conv1D(num_filters, 3, activation='relu')(embedding)
l_cov3 = Conv1D(num_filters, 4, activation='relu')(embedding)

l_pool1 = GlobalMaxPooling1D()(l_cov1)
l_pool2 = GlobalMaxPooling1D()(l_cov2)
l_pool3 = GlobalMaxPooling1D()(l_cov3)

tensors = Concatenate(axis=1)([l_pool1, l_pool2, l_pool3])

dense = Dense(300, activation='relu')(tensors)
dropout = Dropout(0.5)(dense)
flatten = Flatten()(dropout)

preds = Dense(1, activation='sigmoid')(flatten)

checkpoint = ModelCheckpoint('Model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

model = Model(inputs=inputs, outputs=preds)
adam = Adam(lr=1e-5, decay=1e-5)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, tensor_board], validation_data=(X_val, y_val), shuffle=True)
