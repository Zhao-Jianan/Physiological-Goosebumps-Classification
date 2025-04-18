# 📖 User Manual

## 📂 Project File Structure

This project is developed using the **MMACTION2** toolkit. As shown in **Figure 20**, the project consists of several `.ipynb` script files and the `mmaction2` directory.  
To use this system, simply run the script files in order.

![Figure 20](./images/Fig20.png)
**Figure 20: Project File Structure**

### 🔹 mmaction2 Folder Structure  

Pay attention to the following key subdirectories in the `mmaction2` folder:

- **`data` Folder**  
  **Path:** `/mmaction2/data`  
  Place your **preprocessed data** and **label files** here.  
  See **Figure 21** for an example.  
  > ⚠️ Make sure the paths in your label files are correct.  
  Use **`/`** for Linux and **`\`** for Windows.

![Figure 21](./images/Fig21.png)
**Figure 21: Data Folder in Experiments**

> 📝 Note: Due to file size limitations of NESS, the submitted project excludes data files.

- **Configuration Files**  
  **Path:** `/mmaction2/configs/recognition/swin`  
  This folder stores the **model configuration files**.  
  Set your **data paths** and **training parameters** here, as shown in **Figure 22**.

![Figure 22](./images/Fig22.png)
**Figure 22: Configuration File Example**

- **`work_dirs` Folder**  
  **Path:** `/mmaction2/work_dirs`  
  Stores **training logs**, **model checkpoints**, and other related outputs.

---

## 🛠️ Data Preprocessing

As shown in **Figure 23**, run all code cells in the `goosebumps_data_preprocessing.ipynb` file.  
> ⚠️ Don’t forget to set and modify the correct **data paths** before running.

![Figure 23](./images/Fig23.png)
**Figure 23: Data Preprocessing Scripts**

---

## 🎛️ Model Training

To retrain the model, run `main_experiment_training.ipynb`, as shown in **Figure 24**.

![Figure 24](./images/Fig24.png)
**Figure 24: Model Training Script**

---

## 📊 Model Evaluation  

For model evaluation, run `model_evaluation.ipynb` as shown in **Figure 25**.

![Figure 25](./images/Fig25.png)
**Figure 25: Model Evaluation Script**

---

## 🎥 Model Inference and Visualization  

To perform model inference and visualize the results:  
- Run all code cells in `model_inference_and_visualisation.ipynb` sequentially.
- Visualization results will be saved in `/mmaction2/inference_result_visualization_videos`.

> ⚠️ Remember to correctly set the **input and output paths** for videos.

![Figure 26](./images/Fig26.png)
**Figure 26: Inference and Visualization Script**

---

## 📥 Download Links

- **Trained Models:**  
  [🔗 Download Here](https://drive.google.com/drive/folders/1sKRlUEbRWIKDUV1J9KU0HsDGMZcpzdgS?usp=sharing)

- **Inference Result Videos:**  
  [🔗 Download Here](https://drive.google.com/drive/folders/108GxlYUOYss_t-Fw84tlacX4DhEG0FA3?usp=sharing)

---

## ✅ Notes

- Due to file size restrictions, **data and trained models are not included** in the project submission.
- Download the required models and videos using the provided links.
