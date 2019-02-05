import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))

model = load_model("Model.hdf5")

loss, acc = model.evaluate(x=X_test, y=y_test )
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

predictions = model.predict(X_test)
y_pred = []
for ans in predictions:
    if ans > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
