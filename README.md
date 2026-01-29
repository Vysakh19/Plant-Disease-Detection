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

project-root/
├── app.py
├── app_with_gradcam.py
├── gradcam.py
├── models/
│   └── model_latest.pth
├── data/
│   ├── Pepper__bell__Bacterial_spot/
│   ├── Pepper__bell__healthy/
│   ├── Potato__Early_blight/
│   ├── Potato__healthy/
│   ├── Potato__Late_blight/
│   ├── Tomato__Target_Spot/
│   ├── Tomato__Tomato_mosaic_virus/
│   ├── Tomato__Tomato_YellowLeaf_Curl_Virus/
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato_Early_blight/
│   ├── Tomato_healthy/
│   ├── Tomato_Late_blight/
│   ├── Tomato_Leaf_Mold/
│   ├── Tomato_Septoria_leaf_spot/
│   └── Tomato_Spider_mites_Two_spotted_spider_mite/
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
