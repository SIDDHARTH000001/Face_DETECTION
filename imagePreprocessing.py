import cv2
import os
import random
import numpy as np
import shutil
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from PIL import Image

def explore_folder(folder_path):
    print(f'Exploring {os.path.basename(folder_path)}')
    image_shapes = []
    num_images = 0
    num_people = 0
    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)
        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            image = cv2.imread(image_path)
            image_shapes.append(image.shape)
            num_images += 1
        num_people +=1
    print(f'Unique image shapes in: {set(image_shapes)}')
    print(f"Total number of images: {num_images}")
    print(f"Total number of people: {num_people}")
    return set(image_shapes), num_images, num_people


def visualize_sample_images(folder_path):
    num_images = len(os.listdir(folder_path))
    num_rows = (num_images + 4) // 5
    num_cols = min(num_images, 5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))

    for i, image_name in enumerate(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)
        sample_image = cv2.imread(image_path)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        row = i // num_cols
        col = i % num_cols

        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        ax.imshow(sample_image)
        ax.axis('off')

    if num_rows > 1:
        for i in range(num_images, num_rows * num_cols):
            fig.delaxes(axes.flat[i])
    else:
        for i in range(num_images, num_cols):
            fig.delaxes(axes[i])

    plt.suptitle(f'Person ID: {os.path.basename(folder_path)}')
    plt.tight_layout()
    plt.show()


def show_random_image(_path):
    person_id = random.choice(os.listdir(_path))
    folder_path = os.path.join(_path, person_id)
    print(f'Samples from {os.path.basename(_path)}')
    visualize_sample_images(folder_path)


def triplets(folder_paths, max_triplets=7):
    anchor_images = []
    positive_images = []
    negative_images = []

    for person_folder in os.listdir(os.path.join('',folder_paths)):

        images = [os.path.join(person_folder,img) for img in
                  os.listdir(os.path.join(folder_paths,person_folder))]

        num_images = len(images)
        if num_images < 2:
            continue

        random.shuffle(images)

        for _ in range(max(num_images-1, max_triplets)):
            anchor_image = random.choice(images)

            positive_image = random.choice([x for x in images
                                            if x != anchor_image])

            negative_folder = random.choice([x for x in os.listdir(folder_paths)
                                             if x != person_folder])

            negative_image = random.choice([os.path.join(os.path.join(folder_paths,negative_folder), img)
                                            for img in os.listdir(os.path.join(folder_paths,negative_folder))])


            anchor_images.append(os.path.join(folder_paths,anchor_image))
            positive_images.append(os.path.join(folder_paths,positive_image))
            negative_images.append(negative_image)


    return anchor_images, positive_images, negative_images


def split_triplets(anchors,
                   positives,
                   negatives,
                   validation_split=0.2):

    triplets = list(zip(anchors, positives, negatives))

    train_triplets, val_triplets = train_test_split(triplets,
                                                    test_size=validation_split,
                                                    random_state=42)

    return train_triplets, val_triplets


def load_and_preprocess_image(image_path, expand_dims=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    image = Image.fromarray(image)

    return image


def show_triplet(triplet):
    num_rows = 1
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))
    for i in range(3):
        ax = axes[i]
        ax.imshow(triplet[i])
        ax.axis('off')

    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()




def batch_generator(triplets, batch_size=32, augment=True):
    total_triplets = len(triplets)
    random_indices = list(range(total_triplets))
    random.shuffle(random_indices)

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),  # Random translation
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Randomly resize and crop, simulating zoom
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the images
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random', inplace=False),  # Randomly erase rectangular regions
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),
    ])

    for i in range(0, total_triplets, batch_size):
        batch_indices = random_indices[i:i + batch_size]
        batch_triplets = [triplets[j] for j in batch_indices]

        anchor_batch = []
        positive_batch = []
        negative_batch = []

        for triplet in batch_triplets:
            anchor, positive, negative = triplet

            anchor_image = load_and_preprocess_image(anchor)
            positive_image = load_and_preprocess_image(positive)
            negative_image = load_and_preprocess_image(negative)

            if augment:
                anchor_image = transform(anchor_image)
                positive_image = transform(positive_image)
                negative_image = transform(negative_image)

            anchor_batch.append(anchor_image)
            positive_batch.append(positive_image)
            negative_batch.append(negative_image)


        yield [np.array(anchor_batch),
               np.array(positive_batch),
               np.array(negative_batch)]


def visualize_triplets(triplets):
    anchor_batch, positive_batch, negative_batch = triplets
    for i in range(len(anchor_batch)):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Anchor")
        plt.imshow(anchor_batch[i])
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Positive")
        plt.imshow(positive_batch[i])
        plt.axis('off')

        plt.subplot(1, 3, 3)

        plt.title("Negative")
        plt.imshow(negative_batch[i])
        plt.axis('off')

        plt.show()
