# Robotic Behaviour Cloning from Images and Actuator Data

This project implements a behaviour cloning framework to train a robotic agent to imitate expert actions using visual and actuator-based data. It combines deep supervised learning, self-supervised representation learning (via VAEs), and performance analysis using real robotic trajectories from the [CLVR JACO Play Dataset](https://github.com/clvrai/clvr_jaco_play_dataset?tab=readme-ov-file).

---

## Dataset

The dataset consists of multi-modal observations of a robotic arm performing pick-and-place tasks:

- **Visual input**:  
  - `front_cam_ob`: RGB images from a front-facing camera  
  - `mount_cam_ob`: RGB images from a top-mounted camera

- **Actuator states**:  
  - `ee_cartesian_pos_ob`: End-effector position and orientation  
  - `ee_cartesian_vel_ob`: End-effector velocity  
  - `joint_pos_ob`: Gripper open/close state

- **Action labels**:  
  - 3D positional vector for arm movement  
  - Discrete action for gripper: open / close / hold

---

## Project Highlights

### Preprocessing & EDA

- Visual inputs normalized using dataset-specific statistics and resized to 224×224 resolution.
- State vectors (e.g., velocities, joint positions) normalized using min-max or z-score scaling.
- Extensive EDA performed to justify preprocessing steps using pixel intensity and value distribution plots.

### Supervised Behaviour Cloning

- Implemented an end-to-end model architecture combining:
  - CNN-based encoders for visual data
  - MLP encoders for positional and velocity data
  - Fusion via elementwise addition and final prediction through a dense MLP head

- **Loss Function**: Combined `MSELoss` for positional actions and `CrossEntropyLoss` for gripper actions  
- **Results (Baseline)**:
  - **Accuracy**: 79.6% (Validation)
  - **F1 Score**: 0.7358
  - **Val Positional Loss**: 0.00426
  - **Val Gripper Loss**: 0.03461

### Model Tuning Experiments

- **Cosine Annealing Scheduler**: Reduced metric oscillations and improved convergence stability.
- **Learning Rate Adjustment (0.0002)**: Improved F1 score and recall for positional prediction.
- **Grip Class Rebalancing**: Applied class weights in loss function to handle class imbalance in gripper actions, significantly improving recall and reducing grip-specific loss.

---

## Self-Supervised VAE for Representation Learning

- Designed a convolutional **Variational Autoencoder** to learn latent representations of visual observations (without using any action labels).
- Trained the VAE on front camera images only.
- **Convergence Achieved**:
  - Train/val reconstruction losses stabilized
  - KL divergence dropped and remained near-zero
  - Generated reconstructions retained key spatial and object features
- **Conclusion**: The learned latent space was well-regularized and generalised across the dataset, suitable for downstream tasks.

---

## Performance Metrics

Tracked using [Weights & Biases](https://wandb.ai/):

| Metric            | Train     | Validation |
|-------------------|-----------|------------|
| Accuracy          | 0.761     | 0.797      |
| F1 Score          | 0.689     | 0.736      |
| Gripper Loss      | 0.061     | 0.035      |
| Positional Loss   | 0.0013    | 0.0043     |

---

## Key Learnings

- Tackled multi-modal fusion between vision and kinematics.
- Improved model generalisation and training stability via scheduler + weighted loss.
- Understood the importance of loss function balance for joint objectives.
- Validated the power of VAEs for downstream policy learning even without access to action supervision.

---

## Technologies Used

- **Frameworks**: PyTorch, torchvision
- **Libraries**: NumPy, matplotlib, PIL, Weights & Biases (wandb)
- **Concepts**: Behaviour Cloning, Self-supervised Learning, VAE, CNNs, Loss Engineering

---

## Links

- Dataset: [CLVR JACO Play Dataset](https://github.com/clvrai/clvr_jaco_play_dataset?tab=readme-ov-file)
- Model tracking: [W&B Example Run](https://wandb.ai/dionfernandes5-university-college-london-ucl-/cw2_v2)

---

## Author Notes

This project was completed as part of the MSc in AI coursework at UCL. It combines core elements of robot learning, deep supervised learning, and self-supervised representation learning using real-world robotic interaction data.
