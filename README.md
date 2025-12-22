# Mesh2CAD
Automated Conversion of Unstructured Mesh Models to Structured Parametric CAD Models

## Flow Chart
![Flow Chart](images/flow_chart.png)


## Environment Setup

### Launch Fusion 360 Environment

1. Install Fusion 360 (requires educational or commercial account)
2. Install Python dependencies:
   ```
   pip install pytorch torch_geometric numpy scipy psutil requests
   ```
3. Launch Gym server in Fusion 360:
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

Pre-trained models are saved in `train/ckpt/` directory.

## Inference

Navigate to `inference` directory and run the inference script:

```bash
cd inference
python main.py --input /path/to/target.smt --agent gcn --search best
```

Main arguments:
- `--input`: Target B-Rep file or folder
- `--agent`: Agent type, options: `rand`, `gcn`, `mlp` [default: `rand`]
- `--search`: Search strategy, options: `rand`, `beam`, `best` [default: `rand`]
- `--budget`: Number of search steps [default: 100]
- `--launch_gym`: Auto-launch Fusion 360 Gym [default: False]

Results are saved in `inference/log/` directory.