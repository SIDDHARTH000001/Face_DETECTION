import SimeseNet
import imagePreprocessing
import torch
import math
from tqdm import tqdm
import torch.optim as optim


ROOT_DIR = "./dataset/faces/"
train_set = "./dataset/faces/training/"
test_set = "./dataset/faces/testing/"
train_batch_size = 64
train_number_epochs = 100

anchor, positive, negative = imagePreprocessing.triplets(train_set,35)
train_trip , val_trip = imagePreprocessing.split_triplets(anchor, positive , negative,validation_split=.12)


# train_images = [tuple(img for img in triplet )for triplet in train_trip]
# example_triplets = [next(imagePreprocessing.batch_generator(train_images, 7))]
# imagePreprocessing.visualize_triplets(example_triplets[0])
# exit(1)
input_shape = ( 128, 128,3)
embedding_net = SimeseNet.EfficientNetEmbedding(128)
siamese_net = SimeseNet.SiameseNet(embedding_net, input_shape)
siamese_model = SimeseNet.SiameseModel(siamese_net)

#_____________________________________________________________________________
#
# anchor_input ,positive_input, negative_input = torch.Tensor(example_triplets[0][0]),torch.Tensor(example_triplets[0][1]),torch.Tensor(example_triplets[0][2])
# #
# loss  = siamese_model(anchor_input, positive_input, negative_input)
# print(loss)
# # _____________________________________________________________________________
# exit(1)

def train_model(model, train_triplets, epochs, batch_size, val_triplets, patience, delta=0.0001):
    best_val_accuracy = 0
    best_val_loss = float('inf')
    temp_patience = patience
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }
    import torch.nn as nn
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    train_steps_per_epoch = math.ceil(len(train_triplets) / batch_size)
    val_steps_per_epoch = math.ceil(len(val_triplets) / batch_size)

    print(train_steps_per_epoch)
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        train_loss = 0.
        val_loss = 0.

        model.train()
        with tqdm(total=train_steps_per_epoch, desc='Training') as pbar:
            for batch in imagePreprocessing.batch_generator(train_triplets, batch_size=batch_size):
                anchor, positive, negative = batch
                anchor ,positive, negative = torch.Tensor(anchor),torch.Tensor(positive),torch.Tensor(negative)

                optimizer.zero_grad()
                loss = model.training_step(anchor, positive, negative)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.update()
                pbar.set_postfix({'Loss': loss.item()})

        model.eval()
        with torch.no_grad():
            with tqdm(total=val_steps_per_epoch, desc='Validation') as pbar:
                for batch in imagePreprocessing.batch_generator(val_triplets, batch_size=batch_size):
                    anchor, positive, negative = batch
                    anchor ,positive, negative = torch.Tensor(anchor),torch.Tensor(positive),torch.Tensor(negative)
                    loss  = model.validation_step(anchor, positive, negative)

                    val_loss += loss.item()

                    pbar.update()
                    pbar.set_postfix({'Loss': loss.item()})

        train_loss /= train_steps_per_epoch
        val_loss /= val_steps_per_epoch

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'\nTrain Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_bkp.pth')
            temp_patience = patience
        else:
            temp_patience -= 1
            if temp_patience == 0:
                print('Early stopping: Validation loss did not improve.')
                break

    torch.save(model.state_dict(), 'final_model_bkp.pth')

    return model, history


siamese_model, history = train_model(siamese_model,
                                     train_triplets=train_trip,
                                     epochs=75,
                                     batch_size=32,
                                     val_triplets=val_trip,
                                     patience=3)
