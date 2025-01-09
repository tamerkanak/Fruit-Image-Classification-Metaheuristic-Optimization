# Fruit Image Classification with Metaheuristic Optimization

This repository contains code for optimizing the hyperparameters of an Artificial Neural Network (ANN) using metaheuristic optimization algorithms. The project uses the [Fruits 360 dataset](https://www.kaggle.com/moltean/fruits) and explores both Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) for hyperparameter tuning.

## Overview

This project aims to classify different types of fruits using an ANN. We use metaheuristic optimization algorithms to find the best set of hyperparameters for this ANN. The project includes:

- **Data Preparation:** Loading and preprocessing images from the Fruits 360 dataset.
- **ANN Model:** A basic ANN model for classification of images.
- **Hyperparameter Tuning:**
    - Genetic Algorithm (GA): Evolving hyperparameters through selection, crossover, and mutation.
    - Particle Swarm Optimization (PSO): Iteratively optimizing hyperparameters based on individual and global best solutions.
- **Performance Evaluation:** Calculating metrics like accuracy, precision, recall, and F1-score, as well as validation accuracy during the optimization process.
- **Visualization:** Displaying the training and validation accuracy/loss of the final ANN model after metaheuristic optimization.

## Project Structure

├── fruit_mh_opt.py # Main Python script

├── README.md # This README file

- `fruit_mh_opt.py`: This script contains all the code for data loading, preprocessing, ANN model definition, GA, PSO, and evaluation logic.

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd Fruit-Image-Classification-Metaheuristic-Optimization
    ```

2.  **Install required libraries:**

    ```bash
    pip install numpy matplotlib scikit-learn tensorflow opencv-python mealpy
    ```
    or you can use:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set your data paths:**

    Modify the `data_path` variable in `fruit_mh_opt.py` to point to your local directory where the Fruits-360 dataset is located.

    ```python
    data_path = r"C:\\Users\\tamer\\Desktop\\Deep Learning\\fruits-360_dataset_original-size\\fruits-360-original-size"
    ```

4.  **Run the script:**

    ```bash
    python fruit_mh_opt.py
    ```

## Key Findings

- The project demonstrates how metaheuristic optimization algorithms can be used to find optimal hyperparameters for an ANN model.
- Performance comparison between Genetic Algorithm and Particle Swarm Optimization in the context of ANN hyperparameter optimization.
- Evaluation of the trained ANN model after metaheuristic hyperparameter tuning.

## Results

The results of the experiments include:
- The best hyperparameters identified by each algorithm (GA, PSO).
- The corresponding validation accuracy achieved by the tuned ANN model.
- Training and validation accuracy/loss plots for the tuned ANN model.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Feel free to fork this repository and contribute by submitting a pull request.

## Contact

For any questions or feedback, feel free to contact:
[Your Name]
[Your Email]
