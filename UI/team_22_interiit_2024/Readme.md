![Project Landing Page](https://res.cloudinary.com/dqo9rn5lp/image/upload/v1733159971/landing_page_ho80ra.png)

# Project Repository Structure

This repository is structured to streamline data processing, modeling, and UI integration. Below is an overview of each folder and its purpose.

---

## 🚀 Main Application

*(This section is currently empty. Fill in with relevant details about the main application.)*

---

## Data

*(This section is currently empty. Fill in with relevant details about the data processing and structure.)*

---

## Data Processing

*(This section is currently empty. Fill in with relevant details about data processing scripts and workflows.)*

---

## Documentation and Demos

*(This section is currently empty. Fill in with relevant details about documentation, videos, and walkthroughs.)*

---

# Model

### Dataset Structure
- **Source (src):** `[batch_size, 22, 66]`  
  - Includes 6 match parameters and 60 player-specific parameters per player.
- **Target (tgt):** `[batch_size, 22, 66]`  
  - Comprises 15 target parameters expanded to 66 dimensions for alignment.

---

### Model Architecture
The model is based on a **Transformer architecture**, optimized for sequential data. Key components include:

1. **Multi-Head Attention**  
   - Captures relationships between players and contextual match features using multiple attention heads.

2. **Encoder**  
   - Processes the input sequence (match and player-level features) using stacked attention and feedforward layers.

3. **Decoder**  
   - Generates player predictions by applying attention mechanisms to both the target and encoder outputs.

4. **Loss Function**  
   - Uses a weighted Mean Absolute Error (MAE) loss:  
     `loss = weighted_mae_loss(prediction, target, lambda_val=10)`  
     - Weights are derived from **Dream 11 fantasy points** for each performance parameter.  
     - Higher weight parameters (e.g., wickets, boundaries) are prioritized for accurate predictions.

---

### Training Workflow
#### 1. Data Loading and Batching
- Match data is structured into `src` (input) and `tgt` (output) tensors.
- A **DataLoader** is employed to process batches of size 32.
- Tensors are **normalized** for efficient training.

#### 2. Training Loop
- Gradually increases the sequence length of the target (`tgt`) during training.
- Computes the loss at each step using the weighted MAE loss function.
- Updates model parameters using the **Adam optimizer**.

#### 3. Saving the Model
- Saves the trained model weights to a `.pth` file.

---

### Testing Workflow
#### 1. Data Loading and Batching
- Processes match and player data into appropriate tensor structures.

#### 2. Normalization
- Applies the same normalization procedure as in training.

#### 3. Prediction
- Outputs are **unnormalized** to restore the original parameter scales.

---

### Performance Metrics
1. **Mean Absolute Error (MAE):**  
   - Computes the MAE between predicted and actual **total team points**.
2. **Fantasy Points Evaluation:**  
   - Calculates fantasy points based on 14 predicted performance parameters.
   - Sorts players by predicted fantasy points and computes the **top-11 total points**.

--- 

---

## Model Artifacts

*(This section is currently empty. Fill in with relevant details about pre-trained models and artifacts.)*

---

## Out of Sample Data

*(This section is currently empty. Fill in with relevant details about evaluation data and integration workflows.)*

---

## Rest

*(This section is currently empty. Fill in with details about any miscellaneous requirements.)*

---

## UI

### BASE_DIR: `/src/UI/PRODUCT_UI`

#### Key Components:
1. **`model.pt`**: ML Model file.  
2. **`api/ml_model.py`**: ML model integrated with the Web UI.  
3. **`api/agent.py`**: LLM Generative AI integrated feature.  
4. **`api/data`**: Stores data for player-to-identifier mapping and player stats.

---

#### Deployed Site:
The Product UI is deployed and accessible at:  
**[http://172.16.4.2:8080](http://172.16.4.2:8080)**  

The Product UI can be tested and utilized at the above URL.

---

## Start the Application Locally

### Frontend (cd `frontend`)
1. Install dependencies:  
   ```bash
   pnpm i
   # or
   npm i --force
2. Start the development server:
   ``` bash
   pnpm run dev
   # or
   npm run dev

### Backend (cd `backend`)
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Apply database migrations:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
4. Create a superuser:
   ```bash
   python manage.py createsuperuser
5. Enter the following details:
   1) Username: admin
   2) Email: admin@gmail.com
   3) Password: 1234
6. Update player data:
   ```bash
   python manage.py update_players.py
7. Update additional data:
   ```bash
   python manage.py update_data.py
