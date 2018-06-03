from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

model = tf.contrib.keras.models.load_model('my_model.h5')

score = model.evaluate(X_test, Y_test, verbose=0)

Y_pred = model.predict(X_test)

print(Y_test[:5])
print(Y_pred[:5, 0])