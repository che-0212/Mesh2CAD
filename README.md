# Mesh2Sequences
Automated Conversion of Unstructured Mesh Models to Structured Parametric CAD Models

## Flow Chart
![Flow Chart](images/flow_chart.png)


## Environment Setup

### Launch Fusion 360 Environment

1. Install Fusion 360 (requires educational or commercial account)
2. Create a conda virtual environment:
   ```bash
   conda create -n mesh2sequences python=3.9
   conda activate mesh2sequences
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Gym server in Fusion 360:
   - Open Fusion 360
   - Go to Tools > Add-ins > Scripts and Add-ins
   - Add and run the Fusion 360 Gym server

## Training

Navigate to `train/src` directory and run the training script:

```bash
cd train/src
python train.py --dataset /path/to/data/ --split /path/to/train_test.json
```

Main arguments:
- `--dataset`: Path to dataset
- `--split`: Train/test split file
- `--mpn`: Network type, options: `gcn`, `mlp` [default: `gcn`]
- `--epochs`: Number of training epochs [default: 100]
- `--lr`: Learning rate [default: 0.0001]

Trained models are saved in `train/ckpt/` directory.

### Download Model Weights

Trained model weights are available on [Google Drive](https://drive.google.com/drive/folders/1eFpZMAzIylqMp6RghP1bQhwqMpPlpOCS?dmr=1&ec=wgc-drive-hero-goto). Download the model checkpoints and place them in the `train/ckpt/` directory for inference.

## Inference

Navigate to `inference` directory and run the inference script:

```bash
cd inference
python main.py --input /path/to/target.step --agent gcn --search best
```

Main arguments:
- `--input`: Target B-Rep file or folder
- `--agent`: Agent type, options: `gcn`, `mlp` [default: `gcn`]
- `--search`: Search strategy, options: `beam`, `best` [default: `best`]
- `--budget`: Number of search steps [default: 100]
- `--launch_gym`: Auto-launch Fusion 360 Gym [default: False]

Results are saved in `inference/log/` directory.