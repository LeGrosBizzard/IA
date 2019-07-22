import tensorflow as tf


def create_model(images):
    
    model = tf.keras.models.Sequential()
    #on dit: modele de type sequence


    
    #ADD LAYERS dans le modele sequence
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    
    #ReLU laisse passer les valeur positive et 0 pour negative
    #soit 0 si x < 0 ou x si x >= 0

    #softmax permet de sortir des proba

    model_output = model.predict(images[0:1])
    #la ca nous donne une prédiction sur la sortie



    return model_output, model








