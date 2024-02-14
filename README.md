# Fashion-Recommendation-System
The Fashion Recommendation System is a deep learning based project which leverages the power of Convolutional Neural Networks (CNNs) for feature extraction from fashion images.

This project primarily aims to provide similar fashion product recommendations based on an input image. This is achieved by employing a ResNet50 model pre-trained on ImageNet data to extract features from the product images. The recommendation is based on cosine similarity or euclidean distance metrics to identify similar products within the fashion dataset.

# Key Features:
Feature Extraction: Leveraging ResNet50, a pre-trained model on ImageNet, to extract features from product images.

Image Recommendations: Given an input image, the system is able to recommend similar fashion items. The similarity is computed based on the cosine similarity or euclidean distance between the feature vectors of the images.

Success Rate Evaluation: The success rate of the recommendation system is evaluated by measuring the similarity between the features of the recommended items and the input image.

# Project Setup and Execution:
The project requires Python 3.6 or above along with Flask, TensorFlow, NumPy, Pandas, PIL, Scikit-Learn, and Matplotlib libraries.

Clone the repository and navigate to the project directory.

Run the Flask application with the command python app.py.

The Flask server starts on the local machine. Navigate to the home page (http://localhost:5010 by default) using a web browser.

Upload a fashion product image using the upload option.

The system then processes the image, identifies the features, and displays similar fashion products.

# Dataset:
The dataset used in this project comprises of various fashion products and their respective details, including the product image, gender, master category, subcategory, article type, base color, season, year, and usage.

Please note that the system requires the pre-computed image features and a list of image file paths for it to work. These need to be provided in the form of pickle files as seen in the script.

# Note:
This project is a showcase of how machine learning and specifically deep learning techniques can be used in the fashion industry to provide an enhanced shopping experience. The recommendation system is basic and there are several ways to improve and optimize it further, like using a more specific dataset or utilizing more complex models for better accuracy. The system does not include functionalities like product tracking, user authentication, or transaction handling, which are crucial aspects of a real-world fashion e-commerce application.

