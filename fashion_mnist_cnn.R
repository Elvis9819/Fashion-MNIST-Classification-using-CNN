library(keras)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% fashion_mnist

# Normalize the data
x_train <- x_train / 255
x_test <- x_test / 255

# Reshape the data to include the channel dimension
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# Define the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = 'accuracy'
)

# Train the model
model %>% fit(
  x_train, y_train,
  epochs = 5,
  validation_data = list(x_test, y_test)
)

# Save the model
save_model_hdf5(model, "fashion_mnist_cnn.h5")

# Make predictions
sample_images <- x_test[1:2,,,drop = FALSE]
predictions <- model %>% predict(sample_images)

# Display images and predictions
par(mfrow=c(1,2))
for (i in 1:2) {
  image(t(apply(sample_images[i,,,1], 2, rev)), col = gray.colors(255), main = paste("Prediction:", which.max(predictions[i,]) - 1))
}