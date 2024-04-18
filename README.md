[Watch Demo Video](https://youtu.be/gThWBqbo4KA)

## Setup for Python

1. Install Python ([Setup instructions](https://wiki.python.org/moin/BeginnersGuide))

2. Install Python packages

```
pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt
```

## IDE Setup

1. **Download [PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows).**
2. Follow the installation instructions for your operating system.
3. Open the project in PyCharm to begin development.

## Training the Model

1. **Download the data from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).**
2. Only keep folders related to Potatoes.
3. Run Jupyter Notebook in your browser.
4. Open `training.ipynb` in Jupyter Notebook.
5. In cell #3, update the path to the dataset.
6. Run all the cells one by one.
7. Copy the generated model and save it with the version number in the `models` folder.

## Running the API

### Using FastAPI

1. Get inside `api` folder

```bash
cd api
```

2. Run the FastAPI Server using uvicorn

```bash
uvicorn main:app --reload --host 0.0.0.0
```

3. Your API is now running at `0.0.0.0:800

## How to Run the Model

1. **Clone the Repository:**
   git clone https://github.com/VijayShankarSN/Potato-Disease-Prediction.git

3. **Navigate to Project Directory:**
   cd Potato-Disease-Prediction

4. **Run**
   main.py

5. Open Google Chrome or any other web browser.


6. **Access Localhost:**
Enter the following URL in the browser address bar:
[http://localhost:8000](http://localhost:8000)

7. Click on the upload button to select and upload a potato leaf image.

8. After uploading the image, click on the "Predict" button.

9. A pop-up window will appear displaying the results of the prediction.

10. To upload another image, click on the image in the pop-up window to reopen the upload interface.

11. **Repeat Steps 6-9:** 
 Repeat the process to predict diseases for additional images.

## Description

Potato Disease Prediction

Potato Disease Prediction is an innovative project aimed at revolutionizing agricultural practices by leveraging cutting-edge technology to predict and diagnose diseases in potato plants. Employing state-of-the-art machine learning techniques, this project utilizes TensorFlow, Convolutional Neural Networks (CNNs), data augmentation, and TensorFlow Dataset to accurately detect and classify various diseases affecting potato crops.

**Key Features:**

1. **Advanced Machine Learning Algorithms:** The project harnesses the power of TensorFlow, a leading machine learning framework, to develop robust models capable of accurately predicting potato diseases.

2. **Convolutional Neural Networks (CNNs):** CNNs are employed to effectively capture intricate patterns and features within potato plant images, enabling precise disease identification.

3. **Data Augmentation:** Augmenting the dataset enhances model generalization by generating variations of input images, thereby enriching the training process and improving model performance.

4. **FastAPI Backend:** The backend infrastructure is built using FastAPI, a modern web framework for building APIs with Python. FastAPI ensures efficient communication between the frontend and backend components, facilitating seamless data processing and prediction.

5. **Model Optimization:** To enhance efficiency and deploy the model on resource-constrained devices, optimization techniques such as quantization and TensorFlow Lite are implemented, reducing model size while maintaining high accuracy.

6. **HTML Frontend:** The frontend interface is developed using HTML, providing users with an intuitive and user-friendly platform to interact with the application, upload images, and receive disease predictions.

**Project Structure:**

The project is structured into distinct components for seamless integration and modularity:

| Component             | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| Data Collection       | Acquisition and preprocessing of potato plant image dataset   |
| Model Development     | Training CNN models using TensorFlow and data augmentation   |
| Backend Development   | Implementation of FastAPI backend for data processing         |
| Model Optimization    | Optimization techniques like quantization and TensorFlow Lite |
| Frontend Development  | HTML interface for user interaction and result display        |
| Testing and Deployment| Evaluation of model performance and deployment on GitHub      |

**How to Use:**

1. **Upload Image:** Users can upload images of potato plants affected by diseases via the frontend interface.
2. **Prediction:** The uploaded image is processed through the trained CNN model, which predicts the type of disease present.
3. **Display Results:** The frontend displays the prediction results, providing users with actionable insights for disease management and crop protection.

**Conclusion:**

Potato Disease Prediction represents a significant advancement in precision agriculture, offering farmers a reliable tool for early disease detection and mitigation. By harnessing the power of machine learning and web technologies, this project contributes to enhancing crop yield, reducing losses, and promoting sustainable farming practices. Explore the repository on GitHub to learn more and contribute to the future of agricultural innovation.

Inspiration: https://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions
