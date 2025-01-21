# 🌱 Plant Disease Detection System for Sustainable Agriculture

Welcome to the **Plant Disease Detection System** repository! This project leverages Deep Learning to detect plant diseases with the goal of promoting sustainable agriculture by enabling early and accurate identification of plant health issues.

---

## 🚀 Features

- **Deep Learning with CNNs**: Built using Convolutional Neural Networks (CNNs) to classify plant diseases from images.
- **Image Dataset Integration**: Automatically processes and augments datasets for training and validation.
- **User-Friendly**: Easy-to-understand code structure and modular design for extensibility.
- **Sustainable Agriculture Focus**: Designed to help farmers minimize crop loss by identifying diseases early.

---

## 🖼️ Dataset

The project uses a labeled dataset of plant images categorized by disease type, primarily including plants such as apple, blueberry, cherry, corn, grape, orange, peach, potato, strawberry, and tomato. Images are preprocessed into uniform dimensions for model training. If you'd like to use your own dataset, ensure it's structured as follows:

```
Dataset/
|-- train/
|   |-- Class1/
|   |-- Class2/
|   ...
|-- valid/
|   |-- Class1/
|   |-- Class2/
|   ...
```

---

## 🛠️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Abhidongre/Plant-Disease-Detection-System-for-Sustainable-Agriculture.git
   cd Plant-Disease-Detection-System-for-Sustainable-Agriculture
   ```

2. **Install Dependencies**

   Ensure you have Python 3.7+ and the following libraries installed:

   ```bash
   pip install tensorflow matplotlib pandas seaborn
   ```

3. **Prepare the Dataset**

   - Place your dataset in the `Dataset/` directory.
   - Ensure subdirectories are labeled appropriately for each class.

---

## 📚 How to Use

1. **Training the Model**

   Run the Jupyter notebook to train the CNN model:

   ```bash
   jupyter notebook Train_plant_disease.ipynb
   ```

   This will:

   - Load and preprocess the dataset.
   - Train the CNN model on the training dataset.
   - Validate the model on the validation dataset.

2. **Evaluating the Model**

   The notebook includes code to:

   - Display training/validation accuracy and loss.
   - Test the model on new images.

3. **Make Predictions**

   Use the trained model to predict plant disease on new images:

   ```python
   from tensorflow.keras.models import load_model
   model = load_model('path_to_saved_model.h5')
   prediction = model.predict(new_image)
   ```

---

## 📊 Model Performance

### Accuracy

- **Training Accuracy**: \~99%
- **Validation Accuracy**: \~96%

### Metrics

- Confusion Matrix
- Precision, Recall, and F1 Score

Performance plots are included in the notebook for detailed analysis.

---

## 🧑‍💻 Project Structure

```plaintext
Plant-Disease-Detection-System/
|-- Dataset/              # Directory for training/validation data
|-- Train_plant_disease.ipynb  # Main notebook for training and evaluation
|-- requirements.txt      # List of dependencies
|-- README.md             # Project documentation (this file)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Commit your changes and create a pull request.

---

## ✉️ Contact

Have questions or suggestions? Reach out to [Abhidongre](https://github.com/Abhidongre).

---

## ⭐ Acknowledgments

Special thanks to:

- Open datasets used for training.
- TensorFlow and Keras for making deep learning accessible.
- This project was built during the **AICTE Internship on AI: Transformative Learning**, powered by **TechSaksham**—a joint CSR initiative by **Microsoft** and **SAP**. The program provided mentorship, hands-on learning, and a platform to explore real-world AI applications.

If you found this project helpful, don't forget to give it a star ⭐ on GitHub!

