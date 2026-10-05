# Doodle Classifier: Drawing as a New Way to Search

UE24CS352A Machine Learning Mini-Project

Recognizes hand-drawn doodles from Google's QuickDraw dataset, comparing a
Logistic Regression baseline against Convolutional Neural Networks.

**Author:** [Pradyumna Panigrahy], [PES2UG24CS359]
**Author:** [Prajwal], [PES2UG24CS360]

## Dataset
- Google QuickDraw numpy bitmaps (28x28 grayscale)
- 10 classes: banana, hockey stick, squirrel, watermelon, bathtub, laptop, rain, pond, nose, panda
- 10,000 images per class, split 80/10/10 (train/val/test), seed 42

## Models and Results (10 classes)
| Model | Test Accuracy | Training Time |
|---|---|---|
| Logistic Regression (baseline) | 66.37% | 50.3 s |
| CNN V1 (2 conv + 2 maxpool) | **87.90%** | 55.1 s |
| CNN V2 (1 conv + 1 maxpool) | 87.43% | 38.5 s |
| CNN V3 (1 conv, no pooling) | 86.05% | 43.1 s |

CNN training used 15 epochs, Adam optimizer, batch size 128 on a Colab T4 GPU.

## Project Structure
```
├── notebooks/quickdraw_ml_project.ipynb   # data, training, evaluation (run in Colab)
├── models/best_model.keras                # trained CNN V1
├── results/                               # confusion matrices, curves, demo screenshots
├── app.py                                 # Gradio drawing demo
└── requirements.txt
```

## Setup (Python 3.12 recommended)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Demo
```
python app.py
```
Open http://127.0.0.1:7860, draw a doodle, and see the predicted class.

## Retrain
Open `notebooks/quickdraw_ml_project.ipynb` in Google Colab (GPU runtime) and run all cells.