# Part 3 results

## Selected run

The current experiment is `vast_256_seed3143_v1`. It completed all 200 planned
epochs, or 102,400 update pairs. I used a 256 × 256 CycleGAN with two generators,
two PatchGAN discriminators, nine residual blocks per generator, batch size 1,
cycle weight 10, identity weight 5, and seed 3143. Training used every image in
the 300-Monet and 7,038-photo folders as part of the sampling process.

The checkpoint-selection curve improved from FID 143.945067 at epoch 10 to its
minimum of 103.877408 at epoch 190. Epoch 200 was slightly worse at 104.728361,
so I selected epoch 190 using the predeclared lowest-FID rule.

## Submission and evaluation

The saved submission row contains FID 107.21183928701066 and MiFID
0.41853196918964386. Their mean is 53.81518562810015. My latest submission was
reported by the leaderboard as **-53.8151**. Rank and public/private split were
not supplied, so those fields remain blank instead of being guessed.

The notebook's all-photo local diagnostics report FID 88.253025 and MiFID
88.253025. That local MiFID is explicitly provisional because the organizer's
rule was not confirmed in the saved run. It is therefore kept separate from the
submitted row and leaderboard result.

| Directional metric, 300 images per side | Monet → photo | Photo → Monet |
|---|---:|---:|
| FID subset | 102.992678 | 103.394177 |
| KID mean | 0.019805 | 0.013112 |
| Generative precision | 0.643333 | 0.486667 |
| Generative recall | 0.176667 | 0.456667 |
| Cycle L1 | 0.082639 | 0.105190 |
| LPIPS input vs. translation | 0.297815 | 0.392322 |
| Content cosine | 0.858893 | 0.802858 |

The directional metrics show different tradeoffs. Monet-to-photo has higher
precision but much lower recall, suggesting that its outputs fall into a
narrower region of the photo feature space. Photo-to-Monet has better recall but
lower precision and larger perceptual change. These are feature-based
diagnostics, not paired-image accuracy.

## Runtime and stability

The run used an NVIDIA GeForce RTX 4090 with AMP. Training updates took
10,136.57 seconds and the recorded session wall time was 10,634.07 seconds.
Throughput was 10.1020 training pairs per second, or 20.2041 real input images
per second. Peak allocated memory was 9,088.52 MiB. The generators contain
22,756,358 parameters in total and the discriminators contain 5,529,474.

There were zero nonfinite loss events. The whole-run maximum recorded gradient
norm was 507.683228, while the final epoch remained finite. A large isolated
gradient is still worth monitoring; zero NaNs alone does not prove ideal GAN
stability.

## Interpretation and limits

The checkpoint trend supports selecting epoch 190, but it does not prove that
the model generalizes to unseen domains because the supplied image domains are
also used for training and evaluation. The 7,038-image inference manifest shows
that every photo was exported rather than hand-picking successful examples.

The repository includes a 30-sample blinded packet for two independent raters.
The rating sheet is still empty, so I do not report human style, content,
artifact, or agreement scores. Concrete visual limitations are documented in
[failure_analysis.md](failure_analysis.md).
