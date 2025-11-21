<p align="center">
  <img src="assets\header.png" alt="App Header" width="250"/>
</p>

## 🌸 102 Category Flower Classification using ResNet50

This project implements a transfer learning pipeline for classifying flowers into 102 categories using a pretrained ResNet-50 model. The model is fine-tuned on the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), with hyperparameter tuning done using **Optuna** and support for **ONNX export**.

---

### 📌 Project Highlights

* ResNet50 transfer learning (final FC layer fine-tuned)
* Optuna-based hyperparameter tuning with:

  * Learning rate
  * Optimizer (`Adam`, `SGD`)
  * Batch size
  * Weight decay
* Early stopping + Optuna pruning
* Model export to PyTorch `.pth` and ONNX `.onnx` format

---

## Demo 

<p align="left">
  <img src="assets\resnet_demo.gif" alt="App Header" width="600"/>
</p>

---

### 📎 File Structure

```
├── app/
|   ├── __init__.py
|   ├── inference.py
|   ├── main.py
|   ├── preprocessing.py
|   ├── upload.py
|   └── utils/
|       └── class_mapping.json
|
├── assets/
|   └── resnet_demo.gif
|
├── model/
|   ├── best_flower_model.pth   # PyTorch Model
|   └── resnet50_flower.onnx    # ONNX Model
|
├── model-training.ipynb            # Training Notebook (ran on Kaggle)
|
├── requirements.txt
|
└── README.md
      
```

## 🧪 Training Details 
### 📁Dataset

* [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
* Dataset contains:

  * 8,189 flower images
  * 102 classes
* Labels and splits are loaded from `.mat` files

---

### ⚒ Training Workflow

1. Load pretrained ResNet-50 from `torchvision.models`.
2. Replace the final FC layer to output 102 classes.
3. Freeze all layers except FC.
4. Train with `CrossEntropyLoss`, tuned optimizer.
5. Use Optuna to search for the best hyperparameters.
6. Track Validation F1 Score `val_f1` as the objective metric.
7. Apply early stopping and Optuna pruning.
8. Run final training with the best hyperparameters and early stopping.
9. Save the best model.
    
---

### ⚙ Hyperparameters Tuned via Optuna

* lr:  `1e-4, 5e-3`
* optimizer: `['Adam', 'AdamW']`
* batch_size: `16, 32`
* weight_decay: `1e-5, 1e-3`
* dropout: `0.4, 0.6`

---

## 📊 Results & Performance

### Best Hyperparameters:

  ```json
  {
    "optimizer": "Adam",
    "lr": 0.00022715809721755387,
    "batch_size": 16,
    "weight_decay": 1.065704022543824e-05, 
    "dropout": 0.4664370979686941
    }
  ```
### Performance: 
| Metric        | Test     |
| ------------- |----------|
| **Accuracy**  | `91.40%` |
| **Precision** | `90.34%` |
| **Recall**    | `92.97%` |
| **F1 Score**  | `91.04%` |
---


## How to run? 

1. Clone the Repository
```bash
git clone https://github.com/dipesh1dp/toxic-comment-app.git
cd RESNET50-FINE-TUNING
```
2. Create and Activate a Virtual Environment (Optional but Recommended)
On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install all dependencies
```bash
pip install -r requirements.txt
```
4. Run the FastAPI server 
```bash
uvicorn app.main:app --reload 
```

---

### 🙌 Credits

* [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/)
* TorchVision for pretrained models
* Optuna for efficient hyperparameter tuning

---

### 📌 License

MIT License

---

Learning Project by [Dipesh Pandit](https://www.linkedin.com/in/dipesh1dp/).
