from tensorflow.keras.models import load_model
model = load_model('weights/mobilenet_modele.h5')

model.summary()

test_directory = 'DB/test'

test_datagen = ImageDataGenerator(rescale=1./255)
test_batches = test_datagen.flow_from_directory(
        test_directory,
        target_size=(224,224),
        batch_size=1,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=True)

y_pred_raw = model.predict(test_batches)
y_pred = np.argmax(y_pred_raw, axis=1)
y_true = test_batches.classes

[0,     1, 0,   0,   0]
[0.5, 0.1, 0, 0.3, 0.1]

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
# Using 'auto'/'sum_over_batch_size' reduction type.
cce = tf.keras.losses.CategoricalCrossentropy()
cce(y_true, y_pred).numpy()

>>> 0.1

ce_scores = []

for im in test_batches:
    