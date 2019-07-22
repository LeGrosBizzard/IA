def compilation_model(model):
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"]
        )


    #ERREUR = sparse categorial_crossentropy
    #sparse categorial_crossentropy ->

    #1) ca récupere les proba des targets/images
        #et on en fait un vecteur

    #2) on veut une haute confidence de la proba (proche de 1)
    
        #par le softmax on veut une distrib de la proba:
            #si une proba d'une donnée augmente
                #alors les autres proba des autres données baissent

    #3) On prend le log des données de proba:
        #(propriété de convergence) qui minimise l'erreur

    #4) On fait la moyenne du vecteur



    """
    Ex: im[0: 5]
    image1 = cat5 proba[4] = 0.4849848 ....
    vecteur = [0.4849848, 0.04849848,  ... ]
    vecteur = log(vecteur)
    vecteur = mean(vecteur)
    """


    



    #OPTIMIZER = DESCENTE DE GRADIENT(un type) ici stocastique gradient descent
    #METRIC = ACCU = prédiction juste ex sur 100 - 50 = 50%,
    #on compile notre modele


    return model
