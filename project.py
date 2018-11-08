model =  Sequential()
# x = (7049,96,96)
# y = (7049,30)
model.add(Flatten(input_shape= (96,96)))
model.add(Dense(128,activation = "relu"))
model.add(Dropout(0.1))

model.add(Dense(64,activation = "relu"))
model.add(Dense(30))

model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['mae','accuracy'])
model.fit(X_train,y_train,epochs = 50, batch_size = 128,validation_split = 0.2)

loss_and_metrics = model.evaluate(X_test)
model.save('model.h5')

json_string = model.to_json()
model = model_from_json(json_string)
model.load_weights('model.h5',by_name = True)
model.load_model('model.h5')

def show_results(images_index):
  pred = model.predict(X_test[images_index:(images_index+1)])
  show_images(X_test[images_index], pred[0])
