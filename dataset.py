import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#donnée de test

import sys




def importation_dataset():
    
    #on import fashion_mnist
    fashion_mnist = tf.keras.datasets.fashion_mnist

    #on recupere les images et targets
    (images, targets), (_, _) = fashion_mnist.load_data()

    images = images[:1000]
    targets = targets[:1000]
    #1000 image qui font 28*28
    #target = catégorie de l'image de 0 a 9





    images = images.reshape(-1, 784)
    #on applatit l'image (flatten)
    #normalisation reshape les données

    images = images.astype(float)


    scaler = StandardScaler() #<- objet
    #StandardScaler : z = (x  - u)/s

    #variable du pixel - moyenne de toutes les entrées / ecart type
    #-> recentre les valeurs autour de 0:
    #ce qui permet de converger vers une bonne soluce
    #ex :moyenne = 73 -> -1.04
    #ecart type = 90 -> 1.0000007


    images = scaler.fit_transform(images)
    #ce qui nous envoie une nouvelle dataset normaliser,
    #permet de mettre autour de 0
    #ce qui permet de converger vers une bonne soluce
    #ex: im = 255 -> 5.1568489
    #ex: im = 10 -> -5.15641
    #evite les effets echelles
    #StandardScaler



    images_train, images_test, targets_train, target_test =\
                  train_test_split(images,
                                   targets,
                                   test_size=0.2,
                                   random_state=1)

    #prend en parametre les images, target, pourcentage de valeur










    ##print(images.mean())
    ##print(images.std())






    return images_train, targets_train, images






