# Checkpoints

The notebook saves resumable training state and selects the strongest evaluated
epoch using FID on a fixed validation subset. See `src/Part3.ipynb` for the
checkpointing and selection procedure.

## Archived model weights

`generators_256_finetune_epoch_050.pt` contains both CycleGAN generator state
dictionaries from the earlier 256×256 fine-tuning run at epoch 50. It is a
generator-only checkpoint, so optimizer and discriminator states are excluded.

- Image size: 256×256
- Epoch: 50
- Keys: `G_A2B`, `G_B2A`, `config`
- SHA-256: `166c06922fbfadd9ffc63b9d8c2f73e2cfc81d8bc7d8309be47ddd36c4388ef9`
