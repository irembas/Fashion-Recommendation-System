from flask import Flask, render_template, request, redirect
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split



def extract_img_features(img_path, model):
    img = keras_image.load_img(img_path, target_size=(224,224))
    img_array= keras_image.img_to_array(img)
    expanded_img =np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    features =model.predict(preprocessed_img).flatten()
    normalized_features = features/norm(features)
    return normalized_features


def cosine_similarity(vec_a,vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a =norm(vec_a)
    norm_b =norm(vec_b)
    similarity = dot_product/(norm_a * norm_b)
    return similarity


def pearson_similarity(vec_a,vec_b):
    mean_a=np.mean(vec_a)
    mean_b=np.mean(vec_b)
    deviation_a = vec_a-mean_a
    deviation_b= vec_b-mean_b
    numerator = np.sum(deviation_a * deviation_b)
    denominator = norm(deviation_a) * norm(deviation_b)
    similarity = numerator/denominator
    return similarity

def knn_similarity(features,features_list, n_neighbors=5):
    distances = np.array([cosine_similarity(features, feat) for feat in features_list])
    indices =np.argsort(distances)
    knn_indices =indices[:n_neighbors]
    return knn_indices


def recommend(features,features_list, img_files_list, styles_data, metric='cosine',n_recommendations=6,knn_n_neighbors=5):
    if metric == 'cosine':
        similarities = np.array([cosine_similarity(features, feat) for feat in features_list])
        indices = np.argsort(similarities)[::-1]
    elif metric == 'pearson':
        similarities = np.array([pearson_similarity(features, feat) for feat in features_list])
        indices = np.argsort(similarities)[::-1]
    elif metric == 'knn':
        knn_indices = knn_similarity(features, features_list, n_neighbors=knn_n_neighbors)
        indices = knn_indices
    recommended_items = []
    for idx in indices[1:n_recommendations]:
        image_file = img_files_list[idx]
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        style_data = styles_data[styles_data['id'] == int(image_id)]
        if not style_data.empty:
            recommended_items.append({
                'id': int(image_id),
                'productDisplayName': style_data['productDisplayName'].values[0],
                'season': style_data['season'].values[0],
                'year': style_data['year'].values[0],
                'usage': style_data['usage'].values[0],
                'image_path': "static/images/" + os.path.basename(image_file)
            })

    return recommended_items

def print_recommended_items(recommended_items):
    print("Recommended Items:")
    for item in recommended_items:
        print(item)


def compute_success_rates(recommended_items, input_img_path, styles_df, features_to_check, model):
    input_img_features = styles_df[styles_df['image'] == input_img_path][features_to_check].values[0]
    counters = [0] * len(features_to_check)

    for item in recommended_items:
        item_image_path = "/Users/dogagundogar/Desktop/project/" + item['image_path']
        item_features = styles_df[styles_df['image'] ==item_image_path][features_to_check].values[0]
        for i in range(len(features_to_check)):
            if item_features[i] == input_img_features[i]:
                counters[i]+=1

    success_rates=[count/len(recommended_items) for count in counters]
    return success_rates


def compute_overall_success_rate(success_rates):
    return sum(success_rates)/ len(success_rates)


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model = ResNet50(weights="imagenet",include_top=False,input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

features_list = pickle.load(open("/Users/dogagundogar/Desktop/project/image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("/Users/dogagundogar/Desktop/project/img_files.pkl", "rb"))

styles_df = pd.read_csv('/Users/dogagundogar/Desktop/project/styles.csv', error_bad_lines=False)
styles_df.columns = styles_df.columns.str.strip()
styles_df['id'] = styles_df['id'].astype(int)

old_common_path = "/kaggle/input/fashion-product-images-small/"
new_common_path = "/Users/dogagundogar/Desktop/archive/myntradataset/"

updated_img_files_list = [path.replace(old_common_path, new_common_path) for path in img_files_list]

with open("/Users/dogagundogar/Desktop/project/updated_img_files.pkl", "wb") as file:
    pickle.dump(updated_img_files_list, file)
    img_files_list = updated_img_files_list

input_img_path = "/Users/dogagundogar/Desktop/project/static/images/10005.jpg"
assert os.path.exists(input_img_path), f"The file does not exist at {input_img_path}"

input_img_features=extract_img_features(input_img_path, model)


base_path="/Users/dogagundogar/Desktop/project/static/"

features_to_check = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']
styles_df['image'] = styles_df['id'].apply(lambda x: "/Users/dogagundogar/Desktop/project/static/images/" + str(x) + ".jpg")

train_df, test_df = train_test_split(styles_df, test_size=0.001, random_state=42)

success_rates =[]
total_success_rates = 0

for idx, row in test_df.iterrows():
    input_img_features = extract_img_features(row['image'], model)
    recommended_items = recommend(input_img_features, features_list, updated_img_files_list, styles_df, 'cosine', 5)

    success_rate = compute_success_rates(recommended_items, row['image'], styles_df, features_to_check, model)
    average_success_rate = np.mean(success_rate)
    total_success_rates += average_success_rate

overall_success_rate = total_success_rates / len(test_df)
print("Success Rate: ", overall_success_rate)


metrics = ['cosine', 'pearson', 'knn']

recommendations = {}
overall_success_rates = {}

# hepsi için accuracy hesabı
for metric in metrics:
    recommended_items = recommend(input_img_features, features_list, img_files_list, styles_df, metric=metric)
    success_rates = compute_success_rates(recommended_items, input_img_path, styles_df, features_to_check, model)
    overall_success_rate = compute_overall_success_rate(success_rates)
    recommendations[metric] = recommended_items
    overall_success_rates[metric] = overall_success_rate
    print(f"Success Rates for {metric.capitalize()} Recommendations:")
    print(success_rates)
    print(f"Overall Success Rate: {overall_success_rate}")


# en yüksek sonuç alan
best_metric =max(overall_success_rates, key=overall_success_rates.get)

# öneriyi ona göre yapsın
best_recommendations = recommendations[best_metric]
print(f"\nUsing {best_metric.capitalize()} for Recommendations as it has the highest overall success rate.")
print_recommended_items(best_recommendations)



print("\nOverall Success Rates:")
for metric, success_rate in overall_success_rates.items():
    print(f"{metric.capitalize()}: {success_rate}")



@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            input_img_features = extract_img_features(filename, model)
            recommended_items = recommend(input_img_features, features_list, img_files_list, styles_df, metric=best_metric)  # Using the best metric

            return render_template('results.html', uploaded_image=os.path.basename(filename),
                                   recommended_images=recommended_items)

    return render_template('upload.html')


app.run(debug=True, port=5016)