import imagePreprocessing
import SimeseNet
import torch

ROOT_DIR = "./Dataset/faces/"
train_set = "./Dataset/faces/training/"
test_set = "./Dataset/faces/testing/"
train_batch_size = 64
train_number_epochs = 100

anchor, positive, negative = imagePreprocessing.triplets(test_set,32)
train_trip , val_trip = imagePreprocessing.split_triplets(anchor, positive , negative,validation_split=0.001)

train_images = [tuple(img for img in triplet )for triplet in train_trip]

input_shape = ( 128, 128,3)
embedding_net = SimeseNet.EfficientNetEmbedding(128)
siamese_net = SimeseNet.SiameseNet(embedding_net, input_shape)
siamese_model = SimeseNet.SiameseModel(siamese_net)

siamese_model.load_state_dict(torch.load('final_model_bkp.pth'))

acc=[]
for i,batch in enumerate(imagePreprocessing.batch_generator(train_images, 5)):
    if i==5:break
    anchor_input ,positive_input, negative_input = torch.Tensor(batch[0]),torch.Tensor(batch[1]),torch.Tensor(batch[2])
    res1,res2 = siamese_model(anchor_input ,positive_input, negative_input )
    print(res1)
    acc.append(res1.mean().item())

print(sum(acc)/len(acc))



# print(siamese_model(anchor_input ,positive_input, negative_input ))

# output_________________________________________________________
# tensor([0.8398, 0.9550, 0.9471, 0.8737, 0.9310], grad_fn=<SumBackward1>)
# tensor([0.9182, 0.8752, 0.9731, 0.9774, 0.9579], grad_fn=<SumBackward1>)
# tensor([0.8549, 0.9314, 0.9701, 0.9432, 0.9520], grad_fn=<SumBackward1>)
# tensor([0.9380, 0.9778, 0.9723, 0.9907, 0.9186], grad_fn=<SumBackward1>)
# tensor([0.8096, 0.9535, 0.9783, 0.9851, 0.8354], grad_fn=<SumBackward1>)
# 0.9303746223449707
