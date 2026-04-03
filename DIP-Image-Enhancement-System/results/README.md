# DIP Lab 06 – Smart Image Enhancement & Analysis System
**Name:** Muhammad Talal Tariq  |  **Reg ID:** 235154

---

## ▶️  HOW TO RUN ONLINE (Google Colab – FREE, No Install Needed)

### STEP 1 – Open Google Colab
Go to: https://colab.research.google.com
Sign in with your Google account.

---

### STEP 2 – Create a New Notebook
Click  **"New Notebook"**  (top left)

---

### STEP 3 – Install Required Libraries
In the first cell, paste and run:

```python
!pip install opencv-python-headless matplotlib numpy
```
Press **Shift + Enter** to run.

---

### STEP 4 – Upload Your Image
In the second cell, paste and run:

```python
from google.colab import files
uploaded = files.upload()   # a file picker will appear
```
Upload any image (JPG / PNG).  
After upload, note the filename (e.g. `hawkes_bay_in.jpg`).

---

### STEP 5 – Upload the Python Script
In the next cell, paste and run:

```python
from google.colab import files
files.upload()   # upload  lab06_235154.py
```

---

### STEP 6 – Change the Image Name in the Script (if needed)
In a new cell run:

```python
# Replace 'input_image.jpg' with your actual uploaded filename
import subprocess
subprocess.run(["sed", "-i",
                "s/input_image.jpg/hawkes_bay_in.jpg/g",
                "lab06_235154.py"])
print("Done")
```
*(Skip this step if you renamed your image to `input_image.jpg`)*

---

### STEP 7 – Run the Full Application
In a new cell paste and run:

```python
exec(open("lab06_235154.py").read())
```

All 6 phases will run automatically and figures will appear inline.

---

### STEP 8 – Download Output Images
```python
import os, zipfile
with zipfile.ZipFile("lab06_outputs.zip", "w") as z:
    for f in os.listdir("outputs"):
        z.write(os.path.join("outputs", f))
from google.colab import files
files.download("lab06_outputs.zip")
```

---

## 📁 Output Files Generated

| File | Description |
|------|-------------|
| `phase61_acquisition.jpg` | Original + Grayscale + Matrix |
| `phase62a_sampling.jpg` | 5 sampling scales compared |
| `phase62b_quantization.jpg` | 8-bit / 4-bit / 2-bit comparison |
| `phase63a_rotation.jpg` | Rotation at 7 angles |
| `phase63b_trans_shear.jpg` | Translation + Shearing |
| `phase63c_inverse.jpg` | Inverse transformation (restore) |
| `phase64_intensity.jpg` | Negative / Log / Gamma comparison |
| `phase65a_orig_hist.jpg` | Original histogram |
| `phase65b_he_comparison.jpg` | Manual HE vs cv2 HE |
| `phase65c_cdf.jpg` | CDF curve |
| `phase65d_stddev.jpg` | Contrast std-dev bar chart |
| `phase66_final_pipeline.jpg` | Full pipeline result |
| `phase66_hist_before_after.jpg` | Before/After histogram |
| `235154_enhanced_output.jpg` | Final enhanced image |

---

## 🔁 Quick Alternative (Paste Directly into Colab)

If uploads are slow, just paste the entire contents of  
`lab06_235154.py`  into a single Colab cell and press **Shift+Enter**.  
The script auto-downloads a sample image if none is provided.
