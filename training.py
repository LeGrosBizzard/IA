import matplotlib.pyplot as plt


def trainning_model(model, images_train, targets_train):

    
    history = model.fit(images_train, targets_train, epochs=10, validation_split=0.2)
    #fit = on lance l'apprentissage
    #les images, les targets, puis regarde l'erreur
    #on utilise optimizer et on réduit erreur
    #et cela 10 fois sur la dataset


    loss_curve = history.history["loss"]
    acc_curve = history.history["acc"]

    loss_val_curve = history.history["val_loss"]
    acc_val_curve = history.history["val_acc"]

    plt.plot(loss_curve, label="Train")
    plt.plot(loss_val_curve, label="Train")
    plt.legend(loc="upper left")
    plt.title("loss")
    
    plt.show()


    plt.plot(acc_curve, label="Train")
    plt.plot(acc_val_curve, label="Train")
    plt.legend(loc="upper left")
    plt.title("accuracy")
    
    plt.show()




