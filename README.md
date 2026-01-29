# Plant-Disease-Detection
Plant disease detection using CNN and transfer learning (ResNet50) with Grad-CAM explainability and web-based deployment.

---

## Technologies Used
- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Streamlit

---

## Dataset
The training dataset was obtained from Kaggle:

https://www.kaggle.com/datasets/emmarex/plantdisease

Due to GitHub size limitations, the dataset is not included in this repository.

Download the dataset and place it inside the `data/` folder.

---

## Trained Model
The final trained model can be downloaded from Google Drive:

https://drive.google.com/drive/folders/18RXM0z0l16i_GwmJuU3K3LEwUmXCr212?usp=sharing

Place the downloaded model file inside the `models/` folder.

---

## Project Structure
```
project-root/
├── app.py
├── app_with_gradcam.py
├── gradcam.py
├── models/
│   └── model_latest.pth
├── data/
│   ├── Early_Blight/
│   ├── Late_Blight/
│   └── Healthy/
└── README.md
```

- `models/` contains trained model files
- `data/` contains dataset class folders
- Folder names inside `data/` are used as class labels

---

## How to Run


 Run the Streamlit app:
   ```
   python -m streamlit run app.py
   python -m streamlit run app_with_gradcam.py
   ```

---

## License
This project is licensed under the MIT License.
