import io
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def get_confusion_matrix(y_true, y_pred, class_names, normalize='pred'):

    # get the confusion matrix
    cnf = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalize)
    df_cm = pd.DataFrame(cnf, index=class_names, columns=class_names)

    # plot it using seaborn
    plt.figure(figsize = (9,8))
    ax = sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(ax.figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image