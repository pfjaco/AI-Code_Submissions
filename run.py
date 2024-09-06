import torch
import torch.nn as nn
import torch.optim as optim
from data import get_data_loaders
from models import FCNet, ConvNet
import matplotlib.pyplot as plt

### Fixed variables ###
BATCH_SIZE = 1000
SEED = 42
#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setup the device for tensor to be stored

### Setup seeds for deterministic behaviour of computations ###
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
###############################################################

# Load dataset
CIFAR_10_dataset = get_data_loaders(BATCH_SIZE)
    
def train_and_test(model_name, dataset, num_epochs, learning_rate, activation_function_name):
    if model_name == "fcnet":
        model = FCNet(activation_function_name=activation_function_name).to(device)
    elif model_name == "convnet":
        model = ConvNet(activation_function_name=activation_function_name).to(device)
    else:
        raise Exception("No such model. The options are 'fcnet' and 'convnet'.")
    train_loader = dataset[0]
    test_loader = dataset[1]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_tloss = 1000000
    avg_loss = 0
    epoch_num = 0
    for epoch in range(num_epochs):
        print("Epoch: ", (epoch_num + 1))
        running_loss = 0
        last_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        last_loss = running_loss / BATCH_SIZE
        running_loss = 0
        avg_loss += last_loss
        epoch_num += 1

    # testing runs
    train_acc = 0
    total_tlabels = 0
    size = 10000
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, tdata in enumerate(test_loader):
            tinputs, tlabels = tdata
            total_tlabels += 1
            toutputs = model(tinputs)
            test_loss += criterion(toutputs, tlabels).item()
            correct += (toutputs.argmax(1) == tlabels).type(torch.float).sum().item()

    correct /= size
    # Test Accuracy is the total number of correctly labeled photos over the number of Labels?
    test_accuracy = 100 * correct
    
    return model, test_accuracy

def hyperparameters_grid_search(model_name, dataset):
    learning_rate_options = [1e-7, 1e-3, 1]
    activation_function_name_options = ["sigmoid", "relu"]
    best_test_accuracy = 0
    best_hyperparameters = {"learning_rate": None, "activation_function_name": None}
        # TODO: Complete grid search on learning rates and activation functions. Keep the number of epochs to be 5.
    # You can use the following print statements to keep track of the hyperparameter search and finally output the best hyperparameters as well as the

    for learn in learning_rate_options:
        for activation_function in activation_function_name_options:
            print(f"Current hyperparameters: num_epochs=5, learning_rate={learn}, activation_function_name={activation_function}")
            model, test_accuracy = train_and_test(model_name = model_name, dataset= dataset, num_epochs=5, learning_rate=learn, activation_function_name=activation_function)            
            print(f"Current accuracy for test images: {test_accuracy}%")
            best_test_accuracy = max(best_test_accuracy, test_accuracy)
            if(best_test_accuracy == test_accuracy):
                best_hyperparameters['learning_rate'] = learn
                best_hyperparameters['activation_function_name'] = activation_function
    
    print(f"Best test accuracy: {best_test_accuracy}%")
    print("Best hyperparameters:", best_hyperparameters)
    

if __name__ == "__main__":
    # Train and test fully-connected neural network and save the model
    
    model, test_accuracy = train_and_test(model_name="fcnet", dataset=CIFAR_10_dataset, num_epochs=5, learning_rate=1e-3, activation_function_name="relu")
    print(f"FCN Test accuracy is: {test_accuracy}%")
    torch.save(model.state_dict(), "cifar-10-fcn.pt")
    print("FCN model saved.")

    # Plot images with predictions
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    images, labels = next(iter(CIFAR_10_dataset[1])) 
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    fig = plt.figure("Example Predictions", figsize=(12, 5))
    fig.suptitle("Example Predictions")
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        img = images[i] * 0.25 + 0.5     # unnormalize
        img = img.numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(f"Label: {classes[labels[i]]}\nPredicted: {classes[predicted[i]]}")
    plt.show()

    # Train and test convolutional neural network and save the model
    model, test_accuracy = train_and_test(model_name="convnet", dataset=CIFAR_10_dataset, num_epochs=5, learning_rate=1e-3, activation_function_name="relu")
    print(f"CNN Test accuracy is: {test_accuracy}%")
    torch.save(model.state_dict(), "cifar-10-cnn.pt")
    print("CNN model saved.")
    
    # Do hyperparameter search
    hyperparameters_grid_search("convnet", CIFAR_10_dataset)