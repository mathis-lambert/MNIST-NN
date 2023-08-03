# Fully Connected Neural Network
## Architecture : 
### 1. Input Layer (Couche d'entrée) :
    - Nombre de neurones : 784 (correspondant à la taille des images MNIST, 28x28 pixels).
    - Fonction d'activation : Pas de fonction d'activation spécifiée (c'est la couche d'entrée).

### 2. Première Couche Cachée :
    - Nombre de neurones : 10.
    - Fonction d'activation : ReLU (Rectified Linear Unit).

### 3. Deuxième Couche Cachée :
    - Nombre de neurones : 20.
    - Fonction d'activation : ReLU (Rectified Linear Unit).

### 4. Troisième Couche Cachée :
Nombre de neurones : 10.
Fonction d'activation : ReLU (Rectified Linear Unit).

### 5. Couche de Sortie (Output Layer) :
    - Nombre de neurones : 10 (correspondant aux 10 classes pour la classification des chiffres de 0 à 9).
    - Fonction d'activation : Softmax (pour obtenir des probabilités de classification).

Le modèle prend en entrée des images MNIST de 28x28 pixels (784 valeurs de pixels) et les propage à travers trois couches cachées successives de 10, 20 et 10 neurones respectivement, en utilisant la fonction d'activation ReLU pour chaque couche cachée. Enfin, les sorties de la troisième couche cachée sont propagées à travers la couche de sortie composée de 10 neurones, où la fonction d'activation softmax est utilisée pour obtenir des probabilités de classification pour les 10 classes.

C'est ainsi que le modèle traite l'entrée et produit des probabilités pour chaque classe de sortie lorsqu'il est évalué sur de nouvelles images MNIST.