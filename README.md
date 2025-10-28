# Lung and Colon Cancer Classification from Histo-pathological Image Using Machine Learning



## Dataset Overview
The dataset contains color 25,000 images with 5 classes of 5,000 images each. All images are 768 x 768 pixels in size and are in jpeg file format. Dataset can be downloaded as a 1.85 GB zip file LC25000.zip. After unzipping, the main folder lung_colon_image_set contains two subfolders: colon_image_sets and lung_image_sets.

The subfolder colon_image_sets contains two secondary subfolders: colon_aca subfolder with 5000 images of colon adenocarcinomas and colon_n subfolder with 5000 images of benign colonic tissues.

The subfolder lung_image_sets contains three secondary subfolders: lung_aca subfolder with 5000 images of lung adenocarcinomas, lung_scc subfolder with 5000 images of lung squamous cell carcinomas, and lung_n subfolder with 5000 images of benign lung tissues.

5 Classess = 5x5000 = 25000 images

**Citation for the dataset-**

Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019



## Installation
1. Download the dataset from the link : [Dataset Link](https://github.com/tampapath/lung_colon_image_set)

2. You can run via Google Colab for the Training Part or use your local machine to train the model. Approximate time to train the model can be ~3 hours depending on the hardware you are using.

3. Required Libraries includes:
   - Numpy
   - Pandas
   - Keras
   - TensorFlow
   - Flask
   - tqdm and some other libraries

4. Once installed use the web-application to render the Web UI to upload image and classify properly.

## Some Disclaimers
This web application is a prototype system developed for research and educational purposes. It is intended solely for use by qualified pathologists and licensed medical professionals.

The AI models integrated herein are experimental and are provided only as diagnostic assistance tools, not as a substitute for professional medical judgment.

This prototype is not approved for clinical use, and the results should not be relied upon for patient diagnosis, treatment, or medical decision-making by any individual.

The developers and affiliated institutions make no warranties, express or implied, and assume no responsibility or liability for any consequences arising from use or misuse of this system.

By accessing this application, you acknowledge that you are a licensed medical professional and understand that this system is a non-clinical prototype used at your own discretion.
## License

[MIT](https://choosealicense.com/licenses/mit/)
