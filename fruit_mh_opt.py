# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 09:44:36 2024

@author: tamer
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import cv2

def load_images_without_resize(directory):

    images = []
    labels = []
    for fruit_dir in os.listdir(directory):
        fruit_path = os.path.join(directory, fruit_dir)
        if os.path.isdir(fruit_path):
            for img_file in os.listdir(fruit_path):
                img_path = os.path.join(fruit_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)  # Resmi olduğu gibi ekle (boyut korundu)
                    labels.append(fruit_dir)
    return images, labels

def resize_images(images, size):
    return np.array([cv2.resize(img, size) for img in images])

# Veri yolları
data_path = r"C:\\Users\\tamer\\Desktop\\Deep Learning\\fruits-360_dataset_original-size\\fruits-360-original-size"
train_path = os.path.join(data_path, "Training")
validation_path = os.path.join(data_path, "Validation")
test_path = os.path.join(data_path, "Test")

# Resimleri yükleme
X_train_images_original, y_train_labels = load_images_without_resize(train_path)
X_validation_images_original, y_validation_labels = load_images_without_resize(validation_path)
X_test_images_original, y_test_labels = load_images_without_resize(test_path)

# NumPy dizisi oluşturma (boyutlar farklı olduğu için dtype=object)
X_train_images_original = np.array(X_train_images_original, dtype=object)
X_validation_images_original = np.array(X_validation_images_original, dtype=object)
X_test_images_original = np.array(X_test_images_original, dtype=object)

print(f"Train set shape: {X_train_images_original.shape}")
print(f"Validation set shape: {X_validation_images_original.shape}")
print(f"Test set shape: {X_test_images_original.shape}")

# Etiketleri kodla
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_labels)
y_validation = label_encoder.transform(y_validation_labels)
y_test = label_encoder.transform(y_test_labels)

# Verileri yeniden boyutlandır
X_train_images = resize_images(X_train_images_original, (32,32))
X_validation_images = resize_images(X_validation_images_original, (32,32))
X_test_images = resize_images(X_test_images_original, (32,32))

# Normalizasyon
X_train = X_train_images / 255.0
X_validation = X_validation_images / 255.0
X_test = X_test_images / 255.0
y_train_categorical = to_categorical(y_train)
y_validation_categorical = to_categorical(y_validation)
y_test_categorical = to_categorical(y_test)

# ANN Modeli
start_time = time.time()
ann = Sequential([
    Dense(128, activation='relu', input_dim=32 * 32 * 3),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_validation_flat = X_validation.reshape(X_validation.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

ann.fit(X_train_flat, y_train_categorical, validation_data=(X_validation_flat, y_validation_categorical), epochs=5, batch_size=32)

ann_validation_accuracy = ann.evaluate(X_validation_flat, y_validation_categorical, verbose=0)[1]
ann_test_accuracy = ann.evaluate(X_test_flat, y_test_categorical, verbose=0)[1]
    
# Test set tahminleri ve metrikler
y_pred_ann = np.argmax(ann.predict(X_test_flat, verbose=0), axis=1)
precision = precision_score(y_test, y_pred_ann, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_ann, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_ann, average='weighted', zero_division=0)
duration = time.time() - start_time

import random

hyperparameter_space = {
    'learning_rate': [0.0001, 0.1],                 # Continuous
    'batch_size': [16, 32, 64, 128, 256],          # Discrete
    'num_hidden_layers': [1, 5],                   # Integer
    'neurons_per_layer': [16, 128],                # Integer
    'activation_function': ['relu', 'tanh', 'sigmoid'],  # Categorical
    'optimizer': ['sgd', 'adam', 'rmsprop']        # Categorical
}

def initialize_population(pop_size):
    print(f"Initializing population with {pop_size} individuals...")
    population = []
    for i in range(pop_size):
        individual = [
            random.uniform(*hyperparameter_space['learning_rate']),
            random.choice(hyperparameter_space['batch_size']),
            random.randint(*hyperparameter_space['num_hidden_layers']),
            random.randint(*hyperparameter_space['neurons_per_layer']),
            random.choice(hyperparameter_space['activation_function']),
            random.choice(hyperparameter_space['optimizer'])
        ]
        population.append(individual)
        print(f"Initialized Individual {i+1}: {individual}")
    return population

def evaluate_fitness(individual, X_train, y_train, X_validation, y_validation):
    print(f"Evaluating fitness for individual: {individual}")
    learning_rate, batch_size, num_hidden_layers, neurons_per_layer, activation_function, optimizer = individual
    
    # ANN modelini oluştur
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation_function, input_dim=32*32*3))
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation_function))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    
    # Modeli derle ve eğit
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=0)
    
    # Doğrulama seti doğruluğunu döndür
    validation_accuracy = model.evaluate(X_validation, y_validation, verbose=0)[1]
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    return validation_accuracy


def crossover(parent1, parent2):
    print(f"Performing crossover between:\nParent1: {parent1}\nParent2: {parent2}")
    cut_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cut_point] + parent2[cut_point:]
    child2 = parent2[:cut_point] + parent1[cut_point:]
    print(f"Generated Children:\nChild1: {child1}\nChild2: {child2}")
    return child1, child2

def mutate(individual):
    print(f"Mutating individual: {individual}")
    gene_to_mutate = random.randint(0, len(individual) - 1)
    if gene_to_mutate == 0:  # Learning rate
        individual[gene_to_mutate] = random.uniform(*hyperparameter_space['learning_rate'])
    elif gene_to_mutate == 1:  # Batch size
        individual[gene_to_mutate] = random.choice(hyperparameter_space['batch_size'])
    elif gene_to_mutate == 2:  # Num hidden layers
        individual[gene_to_mutate] = random.randint(*hyperparameter_space['num_hidden_layers'])
    elif gene_to_mutate == 3:  # Neurons per layer
        individual[gene_to_mutate] = random.randint(*hyperparameter_space['neurons_per_layer'])
    elif gene_to_mutate == 4:  # Activation function
        individual[gene_to_mutate] = random.choice(hyperparameter_space['activation_function'])
    elif gene_to_mutate == 5:  # Optimizer
        individual[gene_to_mutate] = random.choice(hyperparameter_space['optimizer'])
    print(f"Mutated Individual: {individual}")


def genetic_algorithm(pop_size, num_generations, mutation_prob, crossover_prob):
    print(f"Starting Genetic Algorithm with {num_generations} generations...")
    population = initialize_population(pop_size)
    best_individual = None
    best_fitness = -float('inf')
    
    for generation in range(num_generations):
        print(f"\nGeneration {generation + 1}/{num_generations}")
        fitness_scores = [evaluate_fitness(ind, X_train_flat, y_train_categorical, X_validation_flat, y_validation_categorical)
                          for ind in population]
        
        # En iyi bireyi sakla
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[fitness_scores.index(max_fitness)]
        print(f"Best fitness in generation {generation + 1}: {best_fitness:.4f}")
        
        # Seçilim ve yeni nesil oluşturma
        new_population = []
        for _ in range(pop_size // 2):
            # Turnuva seçimi
            parents = random.sample(population, 5)
            parent1 = max(parents, key=lambda ind: fitness_scores[population.index(ind)])
            parent2 = max(random.sample(population, 5), key=lambda ind: fitness_scores[population.index(ind)])
            
            # Çaprazlama
            if random.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutasyon
            if random.random() < mutation_prob:
                mutate(child1)
            if random.random() < mutation_prob:
                mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Yeni nesli popülasyon yap
        population = new_population
    
    print("\nGenetic Algorithm completed!")
    print(f"Best Individual: {best_individual}")
    print(f"Best Fitness (Accuracy): {best_fitness:.4f}")
    return best_individual, best_fitness


best_individual, best_accuracy = genetic_algorithm(
    pop_size=10, 
    num_generations=10, 
    mutation_prob=0.05, 
    crossover_prob=0.3
)

print("\nFinal Results:")
print("Best Individual:", best_individual)
print("Best Accuracy:", best_accuracy)

from mealpy import PSO
from mealpy.utils.space import FloatVar, IntegerVar, StringVar

# Kategorik parametrelerin doğru değerlere eşleştirilmesi
def decode_activation_function(index):
    return ['relu', 'tanh', 'sigmoid'][int(index)]

def decode_optimizer(index):
    return ['sgd', 'adam', 'rmsprop'][int(index)]

def fitness_function(solution):
    # Parametreleri çözümdeki indekslerden doğru değerlere dönüştür
    learning_rate, batch_size, num_hidden_layers, neurons_per_layer, activation_function_index, optimizer_index = solution
    activation_function = decode_activation_function(activation_function_index)
    optimizer = decode_optimizer(optimizer_index)
    
    # `neurons_per_layer`'ı tam sayıya çevir
    neurons_per_layer = int(neurons_per_layer)
    
    # Diğer parametrelerin doğru türde olduğundan emin olun
    batch_size = int(batch_size)
    num_hidden_layers = int(num_hidden_layers)

    # Değişkenlerle fitness hesaplama: 
    # Burada doğrulama doğruluğu (validation accuracy) doğrudan fitness olarak kullanılacak
    accuracy = evaluate_fitness(
        [learning_rate, batch_size, num_hidden_layers, neurons_per_layer, activation_function, optimizer],
        X_train_flat, y_train_categorical, X_validation_flat, y_validation_categorical
    )

    # Fitness skoru olarak doğruluğu kullanıyoruz, yüksek doğruluk daha yüksek fitness anlamına gelir.
    return accuracy

# Parametreler için bounds tanımlama
bounds = [
    FloatVar(0.0001, 0.1),  # learning_rate (sürekli değişken)
    IntegerVar(16, 256),     # batch_size (kesikli değişken)
    IntegerVar(1, 5),        # num_hidden_layers (kesikli değişken)
    IntegerVar(16, 128),     # neurons_per_layer (kesikli değişken)
    IntegerVar(0, 2),        # activation_function (kategorik, 0, 1, 2 indeksleri)
    IntegerVar(0, 2)         # optimizer (kategorik, 0, 1, 2 indeksleri)
]

problem = {
    "obj_func": fitness_function,
    "bounds": bounds,  # Parametrelerin bounds'ları
    "minmax": "max",    # Maksimize et
    "verbose": True,
}

model = PSO.OriginalPSO(epoch=10, pop_size=10)
g_best = model.solve(problem)

print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")