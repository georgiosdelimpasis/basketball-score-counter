# 🏀 Basketball Detection Model Training Guide

Train a custom YOLO model using your IP camera for improved accuracy!

---

## 📋 Complete Workflow

### Step 1: Capture Training Images (10-15 minutes)

**Goal**: Collect 100-300 images of the basketball in different positions.

```bash
python capture_training_images.py
```

**What to capture:**
- ✅ Basketball at different heights (on ground, mid-air, near hoop)
- ✅ **Ball going THROUGH the hoop** (most important! - capture 50+ of these)
- ✅ Ball near hoop but NOT scoring (rim shots, misses)
- ✅ Different lighting conditions
- ✅ Different camera angles (if you move your phone)
- ✅ Basketball partially visible and fully visible
- ✅ The basketball hoop/rim (helps model understand context)

**Controls:**
- `SPACE` - Capture image manually
- `A` - Toggle auto-capture (captures every ~1 second)
- `Q` - Quit

**Tips:**
- Start with auto-capture (`A`) and move the ball around
- Capture 150-300 images total
- Delete blurry images afterward

---

### Step 2: Label Your Images (30-60 minutes)

You need to draw bounding boxes around the basketball in each image.

#### Using Custom Labeling Tool (Recommended - No crashes!)

**1. Launch the labeling tool:**
```bash
python label_images.py
```

**2. How to use:**

A window will open showing your first image.

**Select class first (before drawing):**
- Press **`1`** for `ball` (basketball anywhere)
- Press **`2`** for `ball_scoring` ⭐ (ball going through hoop - MOST IMPORTANT!)
- Press **`3`** for `hoop` (basketball rim/hoop)

**Draw bounding box:**
- **Click and drag** with mouse to draw box around object
- Box will be colored based on selected class (green=ball, yellow=ball_scoring, blue=hoop)
- Press **`Del`** to delete last box if you made a mistake

**Navigate:**
- Press **`S`** to save labels and go to next image
- Press **`D`** to next image (without saving)
- Press **`A`** to previous image
- Press **`Q`** to quit

**3. Label all images:**

For each image, label the objects:

   **`ball`** (Press `1` first) - Basketball anywhere EXCEPT going through hoop
   - Press `1` to select 'ball' class
   - Click and drag around basketball
   - Box will be GREEN

   **`ball_scoring`** (Press `2` first) - Ball ACTIVELY going through hoop ⭐
   - Press `2` to select 'ball_scoring' class
   - Click and drag around basketball (while going through)
   - Box will be YELLOW
   - This is THE MOST IMPORTANT class!

   **`hoop`** (Press `3` first) - Basketball rim/hoop (optional)
   - Press `3` to select 'hoop' class
   - Click and drag around entire hoop/rim
   - Box will be BLUE

**4. Progress:**
- Status bar shows: current image number, selected class, and box count
- Labels auto-save when you press `S`
- You can go back to previous images to fix labels

---

#### Alternative: Using labelImg (if you prefer)

**Note**: labelImg may crash on macOS due to Qt issues. Use the custom tool above if you experience crashes.

**1. Install labelImg:**
```bash
pip install labelImg
```

**2. Launch labelImg:**
```bash
labelImg
```

**3. Setup for YOLO format:**
- Click **"Open Dir"** → Select `training_data/images/`
- Click **"Change Save Dir"** → Select `training_data/labels/`
- Click **"YOLO"** button (top left) to switch to YOLO format
- Make sure it says "YOLO" not "PascalVOC"

**Navigation shortcuts:**
- `D` - Next image
- `A` - Previous image
- `W` - Create box
- `Del` - Delete box
- `Ctrl+S` - Save

**6. Split into train/val:**

After labeling all images, run the split script:

```bash
python split_dataset.py
```

This will automatically:
- Split images 80% train, 20% validation
- Move images to `training_data/images/train/` and `training_data/images/val/`
- Move labels to `training_data/labels/train/` and `training_data/labels/val/`

**Manual split alternative:**
```python
import os
import random
from pathlib import Path
import shutil

# Source directories
img_src = Path('training_data/images')
lbl_src = Path('training_data/labels')

# Get all images
images = list(img_src.glob('*.jpg')) + list(img_src.glob('*.png'))
random.shuffle(images)

# 80-20 split
split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# Create directories
for split in ['train', 'val']:
    (img_src / split).mkdir(exist_ok=True)
    (lbl_src / split).mkdir(exist_ok=True)

# Move files
for img in train_imgs:
    shutil.move(str(img), str(img_src / 'train' / img.name))
    lbl = lbl_src / f"{img.stem}.txt"
    if lbl.exists():
        shutil.move(str(lbl), str(lbl_src / 'train' / lbl.name))

for img in val_imgs:
    shutil.move(str(img), str(img_src / 'val' / img.name))
    lbl = lbl_src / f"{img.stem}.txt"
    if lbl.exists():
        shutil.move(str(lbl), str(lbl_src / 'val' / lbl.name))

print(f"✅ Split complete: {len(train_imgs)} train, {len(val_imgs)} val")
```


---

### Step 3: Train the Model (1-3 hours)

```bash
python train_model.py
```

**What happens:**
- Validates your dataset structure
- Trains YOLOv8 nano model on your basketball images
- Uses Apple Silicon GPU (MPS) for speed
- Saves checkpoints every 10 epochs
- Stops early if no improvement (saves time)

**Monitor progress:**
- Watch the terminal for loss/metrics
- Training plots saved to: `runs/train/basketball_custom/`

**Expected time:**
- 100 epochs × ~30 seconds = ~50 minutes (depends on dataset size)

**Training output:**
```
runs/train/basketball_custom/
├── weights/
│   ├── best.pt       ← Your best model!
│   └── last.pt       ← Last checkpoint
├── results.png       ← Training curves
└── confusion_matrix.png
```

---

### Step 4: Use Your New Model

**Option A: Replace existing model**

```bash
# Backup old model
cp runs/detect/my_ball/weights/best.pt runs/detect/my_ball/weights/best_old.pt

# Copy new model
cp runs/train/basketball_custom/weights/best.pt runs/detect/my_ball/weights/best.pt
```

**Option B: Add as new model** (Update `config/settings.py`)

Add to `AVAILABLE_MODELS`:
```python
"Custom Ball v2 (IP Camera)": {
    "file": "runs/train/basketball_custom/weights/best.pt",
    "size": "18 MB",
    "speed": "Fastest",
    "map": "Custom",
    "description": "Trained on IP camera footage",
    "is_custom": True
},
```

---

## 🎯 Training Tips

### Dataset Quality > Quantity
- **100-200 good images** > 500 poor images
- Focus on variety: different positions, angles, lighting

### Labeling Tips

#### Critical: `ball` vs `ball_scoring`

**Label as `ball_scoring` when:**
- ✅ Ball is passing through the rim/net (even partially)
- ✅ Ball is inside the hoop opening
- ✅ Ball is clearly going downward through the net
- ✅ Any part of ball is below the rim AND inside the hoop

**Label as `ball` when:**
- ❌ Ball is above the hoop (hasn't entered yet)
- ❌ Ball hits the rim and bounces off
- ❌ Ball is beside/under the hoop but not going through
- ❌ Ball is anywhere else on the court

#### General Labeling Rules
- **Tight boxes**: Box should closely fit the ball/hoop
- **Complete ball**: If ball is partially visible, still box the visible part
- **Consistency**: Be consistent with your box sizes
- **When in doubt**: If ball looks like it's scoring → `ball_scoring`

#### Class Distribution (aim for):
- 40% `ball` (regular basketball)
- 40% `ball_scoring` (THE MONEY SHOT - most important!)
- 20% `hoop` (context for the model)

### Training Settings

**If training is too slow:**
```python
# In train_model.py, reduce:
batch_size=8  # Instead of 16
epochs=50     # Instead of 100
```

**If you get memory errors:**
```python
batch_size=4  # Smaller batches
img_size=320  # Smaller images
```

**For better accuracy (slower training):**
```python
model = YOLO('yolov8s.pt')  # Small model instead of nano
epochs=200
```

---

## 📊 Evaluate Your Model

After training, check:

1. **Training curves** (`runs/train/basketball_custom/results.png`)
   - Loss should decrease
   - mAP should increase

2. **Confusion matrix** (`confusion_matrix.png`)
   - Should show high detection rate for "ball"

3. **Test in app**
   - Run the app and test with real basketball
   - Check detection confidence and accuracy

---

## 🔧 Troubleshooting

### "Dataset structure invalid"
- Check folder structure matches expected layout
- Ensure .txt label files match image names
- Verify train/val split exists

### "CUDA out of memory" or "MPS error"
- Reduce `batch_size` to 8 or 4
- Reduce `img_size` to 320

### "Poor detection after training"
- Need more training images (aim for 200+)
- Add more variety (different positions, lighting)
- Train longer (increase `epochs` to 200)
- Check labels are accurate

### "Training loss not decreasing"
- Check labels are correct
- Verify images aren't blurry
- Try different learning rate: `lr0=0.001`

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Capture images
python capture_training_images.py
# Press 'A' for auto-capture, collect 150-200 images

# 2. Label images
python label_images.py
# Press 1/2/3 to select class
# Click+drag to draw boxes
# Press 'S' to save and next
# Label: ball, ball_scoring (IMPORTANT!), hoop

# 3. Split into train/val
python split_dataset.py

# 4. Train
python train_model.py

# 5. Use new model
cp runs/train/basketball_custom/weights/best.pt runs/detect/my_ball/weights/best.pt

# 6. Restart app
streamlit run app.py
```

---

## 📚 Resources

- **Roboflow**: https://roboflow.com (easiest labeling)
- **LabelImg**: https://github.com/heartexlabs/labelImg (free, offline)
- **YOLO Docs**: https://docs.ultralytics.com
- **Training Guide**: https://docs.ultralytics.com/modes/train/

---

**Good luck with training! 🏀**
