# Part 3 GPU rental comparison

Research checked 2026-09-29. Public-provider research was read-only; subsequent Vast offer selection and rental creation reported by the main task are identified separately below. Rates are advertised GPU/server rates, not measured CycleGAN cost. USD conversions use approximately CNY 6.7077 per USD, as checked in the main task; payment conversion fees may differ. [Exchange-rate quote](https://www.bloomberglinea.com/english/quote/USDCNY%3ACUR/).

## Recommendation

The main task selected a US Vast RTX 5090 offer at **$0.461/hour including 50GB disk**, replacing the old $0.681/hour reference in the comparison. This uses the already-funded account and the same GPU model as our prior CUDA workload. It is a practical selection based on the known workload and the marketplace's DLPerf-per-dollar proxy, not proof of the cheapest measured CycleGAN run. A **2–3 hour planning estimate costs about $0.92–$1.38 plus bandwidth** at the quoted rate; the new host has not yet supplied a measured runtime.

Cheaper hourly candidates remain: the live Vast market showed A4000 and 3090 offers, while Hyperstack's A4000 is about $0.157/hour including IP and CloudRift advertises a $0.39/hour 4090. Their relative CycleGAN throughput and memory fit are unmeasured. AutoDL's advertised 5090 is about $0.414/hour and Featurize's about $0.447/hour, but Chinese payment access remains unverified. Compared with the new Vast quote, their same-duration savings are small.

### Selected live Vast offer

The main task observed [Vast's marketplace](https://cloud.vast.ai/) and supplied this account-specific snapshot:

| Field | Observed value |
|---|---|
| Offer / host / machine | 52705881 / 483833 / 140734 |
| GPU and location | 1×RTX 5090 32GB, United States |
| Hourly total | $0.461 = $0.400 GPU + $0.061 for 50GB disk |
| CPU / system RAM | 24 threads / 64GB |
| PCIe / listed bandwidth | PCIe 4.0 ×16 / 22.7GB/s |
| Listed disk / network | NVMe 2,190MB/s / 317 and 295Mbps |
| Marketplace reliability / CUDA ceiling | 99.66% / maximum CUDA 13.0 |
| Marketplace DLPerf | 190.9; synthetic marketplace proxy, not this experiment's speed |
| Transfer rates shown | $17.334/TB and $16/TB; direction labels were not visible, so not assigned to upload/download |
| Account balance at selection | $3.59 |
| Operational state at research handoff | Rental creation reported; running state, SSH health, and training start not yet confirmed |

Other live marketplace offer rates reported by the main task were A4000 $0.108/hour, 3090 $0.193, 4090 $0.409, RTX 6000 Ada $0.641, A100 $0.614, L40S $0.810, and H100 $2.63. These are observed offers, not benchmark results; their detailed resource and transfer terms were not independently audited in this research. Marketplace inventory and prices can change. All account observations here came from the main task rather than this public-source research agent.

## Worldwide shortlist

These are six candidates for this specific SSH/PyTorch job, not an unconditional ranking. Catalog prices and observed stock are deliberately separate. Hourly rates exclude applicable tax or payment conversion fees unless stated otherwise.

| Provider | Relevant hourly price | Extra cost / initial funding | Availability and practical fit |
|---|---|---|---|
| Vast.ai | Selected US 5090: $0.461 including 50GB disk | Balance $3.59 at selection; bandwidth extra | Live offer 52705881 selected; rental creation reported, running/training not yet confirmed at this snapshot |
| Hyperstack | A4000 16GB: $0.15; A6000 48GB: $0.50 | Public IP $0.00672043/hour; additional SSV storage $0.000096774/GB/hour; $5 minimum credit purchase | Catalog, stock unverified. Documented single-A4000 Norway flavor has 4 CPUs, 21GB RAM, 100GB root disk; SSH VM suitable for custom training |
| CloudRift | 4090: $0.39; 5090: $0.60 | Listed local storage/network included; per-second, no minimum rental spend; minimum credit top-up not established | Catalog, not a confirmed bookable offer. Homepage showed 5090 out of stock; user's earlier 4090 availability problem remains unresolved |
| TensorDock | 3090: from $0.23; 4090: from $0.50; 5090: from $0.60 | $5 minimum deposit; CPU/RAM/SSD affect final quote; no bandwidth charges | Console GPU catalog only: live hosts stayed at Loading. Default 2vCPU/4GB/20GB is inadequate for the offered PyTorch image, which requires at least 40GB disk |
| AutoDL | 3090: about $0.197; 4090: $0.280; 5090: $0.414 | WeChat Pay/Alipay/bank transfer; extra storage can cost more; US-card payment not verified | Official advertised prices; stock unverified; +1 registration, SSH, modern CUDA images documented |
| Featurize | 3090: about $0.242; 4090: $0.279; 5090: $0.447 | WeChat Pay/Alipay; US-payment access/minimum deposit unverified | Public page showed 1×4090 and 4×5090; 3090 sold out. Counts are a snapshot, not a reservation |

Sources: [Hyperstack prices](https://www.hyperstack.cloud/gpu-pricing), [Hyperstack flavors](https://docs.hyperstack.cloud/docs/hardware/flavors/), [Hyperstack September 17 minimum-credit change](https://docs.hyperstack.cloud/docs/release-notes/hyperstack/), [CloudRift prices](https://www.cloudrift.ai/pricing), [CloudRift homepage stock](https://www.cloudrift.ai/), [TensorDock deployment catalog](https://console.tensordock.com/deploy), [TensorDock platform/billing](https://www.tensordock.com/). Chinese-provider sources and access details appear below. TensorDock's homepage advertises a 4090 starting at $0.35, but its deployment catalog returned $0.50; do not budget using the lower marketing number without a selectable host quote.

Hyperstack accepts credit/debit cards through Stripe with 3D Secure. Its documentation says a VM in **SHUTOFF still incurs full resource charges**; hibernate or delete to release compute, then account for retained storage/IP. SSH and custom software are supported. [Billing policies](https://docs.hyperstack.cloud/docs/billing/billing-policies/), [FAQ and shutdown behavior](https://docs.hyperstack.cloud/docs/faq/). TensorDock supports root KVM VMs and SSH; exhaustion of prepaid balance can delete the server. [TensorDock FAQ](https://www.tensordock.com/).

### Other providers checked

| Provider | Verified public result | Assessment for Part 3 |
|---|---|---|
| RunPod | Rendered Pod catalog: A5000 $0.27/hour, 3090 $0.50, 4090 $0.74, 5090 $0.99. Running container/volume disk $0.10/GB/month; stopped volume $0.20/GB/month. Starts with $10 credit. | A5000 is a reasonable lower-cost alternative to benchmark; displayed 4090/5090 rates do not beat the alternatives here. Cloud tier and inventory need checkout verification. [Pricing](https://www.runpod.io/pricing), [billing documentation](https://github.com/runpod/docs/blob/main/accounts-billing/billing.mdx) |
| GPU Mart | A4000 16GB Linux $0.25/hour with 24vCPU, 28GB RAM, 320GB SSD, 300Mbps; GPU/CPU/RAM/storage/bandwidth included. Minimum deposit not established. | More CPU/RAM than Hyperstack's documented A4000 flavor, higher hourly cost. **Pausing does not stop charges; destruction required.** Current hourly page does not substantiate an hourly 4090 price. [Official hourly plans](https://www.gpu-mart.com/pricing-hourly) |
| Clore.ai | SSH/Docker marketplace, cryptocurrency funding. Current changelog includes BTC/USDT/USDC/CLORE rentals; public marketplace did not expose a usable current price. | Cannot rank without a current offer including platform/network fees. Older FAQ payment details conflict with newer changelog. [Marketplace](https://clore.ai/marketplace), [FAQ](https://docs.clore.ai/help/faq), [changelog](https://www.clore.ai/changelog) |
| Lambda | Single GPU: A6000 $1.09/hour, A10 $1.29, A100 40GB $1.99, H100 PCIe $3.29, plus applicable tax. | Higher published costs and excess capacity for this small batch-1 job; no measured speed advantage that offsets price. [Official pricing](https://lambda.ai/pricing) |
| RunDiffusion | Managed creative applications and specialized training workflows; general unrestricted SSH/PyTorch training was not established. | Not a direct replacement for this custom repository/CLI workflow. [FAQ](https://www.rundiffusion.com/faq), [terminal documentation](https://www.rundiffusion.com/creators-club-documentation) |

CUDA/PyTorch access is necessary but does not itself verify this code's compatibility. On any new host, check GPU identity, driver, CUDA/PyTorch build, native BF16 support, disk space and a short identical warm-start benchmark before a full run. In particular, a 5090 needs a Blackwell-compatible software stack; an older advertised CUDA template is insufficient evidence. A4000/3090/4090/5090 are distinct NVIDIA models; cheaper V100/Turing entries are not interchangeable with the existing native-BF16 recipe.

## Chinese-provider public prices

| Provider | GPU | CNY/hour | Approx. USD/hour | What was verified |
|---|---|---:|---:|---|
| AutoDL | RTX 5090 32GB | 2.78 | 0.414 | Official advertised rate; actual host inventory/checkout unverified |
| AutoDL | RTX 4090 24GB | 1.88 | 0.280 | Official advertised rate; actual host inventory/checkout unverified |
| AutoDL | RTX 3090 24GB | 1.32 | 0.197 | Official advertised rate; actual host inventory/checkout unverified |
| Featurize | RTX 5090 32GB | 3.00 | 0.447 | Public availability page displayed 4 GPUs |
| Featurize | RTX 4090 24GB | 1.87 | 0.279 | Public availability page displayed 1 GPU |
| Featurize | RTX 3090 24GB | 1.62 | 0.242 | Public page displayed sold out |
| MatPool | RTX 4090 24GB | from 1.54 | from 0.230 | Homepage minimum; offer/discount conditions unverified |
| MatPool | RTX 3090 24GB | from 1.29 | from 0.192 | Homepage minimum; offer/discount conditions unverified |
| MatPool | RTX 5090 32GB | 2.99 | 0.446 | Market banner says summer promotion; continued eligibility unverified |
| MatPool | A100 80GB | from 5.60 | from 0.835 | Homepage minimum; offer/discount conditions unverified |
| Gongji / Suanli | RTX 4090 24GB | 1.98 | 0.295 | Published on-demand cloud/job price; inventory unverified |
| Gongji / Suanli | RTX 5090 32GB | 3.25 | 0.485 | Published on-demand cloud/job price; inventory unverified |
| Gongji / Suanli | RTX 4090 spot job | 1.19 | 0.177 | Published interruptible batch-job price, not equivalent to persistent on-demand SSH |
| Gongji / Suanli | RTX 5090 spot job | 1.95 | 0.291 | Published interruptible batch-job price, not equivalent to persistent on-demand SSH |
| Selected Vast offer | RTX 5090 32GB | — | 0.461 | Live US offer, includes 50GB disk; bandwidth extra |

Sources: [AutoDL pricing on official home](https://www.autodl.com/home?channel=00000000000000000000000040000001&version=7.60.5.106), [Featurize available instances](https://featurize.cn/vm/available), [MatPool home](https://www.matpool.com/), [MatPool market promotion](https://www.matpool.com/host-market/gpu), [Suanli price list](https://www.suanli.cn/price/). AutoDL's home returned a JavaScript-only body when opened directly; the official page's search-indexed content supplied the price table. MatPool's market did not expose selectable host offers to this public text reader. Availability can change before checkout.

No current, bookable single-GPU H100 price was established from these providers. AutoDL instead advertises H800 at CNY 9.98/hour and A800 80GB at CNY 5.59/hour; these are different models, not substitutes in a price table labeled H100/A100. GPUEater's site did not yield verifiable current pricing and is excluded. [AutoDL official catalog](https://www.autodl.com/home?channel=00000000000000000000000040000001&version=7.60.5.106).

## Access, billing, and portability

| Provider | US-user access/payment | Billing and persistence | Training access |
|---|---|---|---|
| AutoDL | Registration explicitly lists +1 US; KYC documentation accepts overseas passports/non-China driving licenses. Payment docs list WeChat Pay, Alipay, bank transfer; PayPal is described as coming soon, not available. US card checkout unverified. | GPU wall time from start to stop, seconds, CNY0.01 minimum; paid disk expansion separate. 30GB system disk, data disk from 50GB; shared file storage 20GB free then CNY0.01/GB/day. Regional shared bandwidth/traffic not separately billed. | SSH and keys; CUDA/PyTorch images, including PyTorch2.8/CUDA12.8 documented. GPUs are dedicated to the instance. Container instances do not support nested Docker. |
| Featurize | Phone or WeChat signup; WeChat Pay/Alipay funding. US phone/cardless-payment eligibility not established. | Per-minute on-demand billing after rental. Return deletes local disk data; copy to cloud storage/local backup first. Homepage advertises30GB free cloud storage. No separate bandwidth price established. | SSH, VSCode/PyCharm, Jupyter; prepared ML environments. Specific current 5090 PyTorch/driver image still needs verification. |
| MatPool | Phone login visible. US-phone eligibility, payment options, minimum top-up not verified. | Per-minute billing advertised; FAQ bills running state only, ends after stop/release. Minimum rates may have conditions. Storage/bandwidth terms not verified here. | SSH documented; CUDA/PyTorch images and Docker-capable instances documented. Public image list includes older environments, so verify 5090-compatible image before rental. |
| Suanli | Signup/payment/US access not verified. | Per-second cloud/on-demand pricing; cheaper spot jobs are explicitly interruptible. Storage and bandwidth must be checked at checkout. | Cloud-host and batch products have different workflows; SSH suitability for this exact product not verified. |

AutoDL sources: [registration](https://autodl.com/register), [identity verification](https://www.autodl.com/docs/real_name_cert/), [payment and billing](https://api.autodl.com/docs/price/), [storage layout](https://api.autodl.com/docs/env/), [shared file-storage charges](https://www.autodl.com/docs/fs/), [network](https://www.autodl.com/docs/network/), [SSH](https://api.autodl.com/docs/ssh/), [framework images](https://www.autodl.com/docs/base_config/), [dedicated GPU allocation](https://www.autodl.com/docs/gpu/).

Featurize sources: [account/payment](https://docs.featurize.cn/docs/user/basic), [rental billing and return](https://docs.featurize.cn/docs/manual/instance-rent), [SSH/IDE access](https://docs.featurize.cn/docs/manual/instance-connection), [free cloud storage](https://featurize.cn/), [maintenance/fault policy](https://docs.featurize.cn/docs/user/maintain). Scheduled maintenance can stop instances, with advance notice; this is not a guarantee against interruptions.

MatPool sources: [login](https://matpool.com/login), [billing FAQ](https://matgo.cn/supports/reference/faqs/), [SSH](https://matpool.com/learn/article/local-ssh-connect-to-matpool-server/), [images](https://www.matpool.com/supports/doc-images/), [Docker support](https://matpool.com/supports/doc-use-docker-on-matpool/). Suanli: [on-demand versus spot pricing](https://www.suanli.cn/price/).

## Speed and total cost

These are reference GPU hardware limits, independent of rental provider. BF16 values below mean **dense Tensor Core BF16 with FP32 accumulation**. FP4 AI TOPS, sparsity-adjusted numbers, FP32 shader TFLOPS, and BF16 throughput must not be treated as interchangeable.

| GPU | CUDA cores | FP32 TFLOPS | Dense BF16/FP32-accumulation TFLOPS |
|---|---:|---:|---:|
| RTX 3090 | 10,496 | 35.6 | 71.2 |
| RTX 4090 | 16,384 | 82.6 | 165.2 |
| RTX 5090 | 21,760 | 104.8 | 209.5 |

Source: NVIDIA RTX Blackwell architecture whitepaper, Appendix A. [Official NVIDIA specifications](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

TFLOPS are not an end-to-end CycleGAN benchmark. Our batch-1 workload also spends time in convolution selection, instance normalization, data loading, CPU work, checkpoint saves, and evaluation. Provider power limits and CPU/disk allocation can alter runtime even for the same GPU model. A short identical benchmark is necessary to choose by **dollars per completed update**.

For illustration only, if exactly the same 5090 work occupies three billed hours, the selected Vast host's GPU plus disk costs about $1.38, versus $1.24 on advertised AutoDL, $1.34 on Featurize, and $1.34 on MatPool's unverified promotion. This same-duration scenario leaves only about $0.04–$0.14 in advertised savings before transfer, setup, storage, and payment differences; it is not a measured runtime comparison.

Within AutoDL's advertised rates, a 4090 must finish in less than 1.48 times the 5090 runtime to cost less (2.78/1.88). Against the selected $0.461/hour Vast 5090, a $0.280/hour 4090 could take up to 1.65 times as long before the displayed hourly charges break even. Neither ratio predicts actual speed. For a brief continuation with credit already on Vast, setup and payment friction can outweigh the small absolute saving.

### Worldwide break-even calculation

For a fixed completed experiment, total consumption is approximately `hourly resource rate × (setup + training + evaluation + export hours) + retained storage + traffic + transaction fees`. New minimum deposits are cash paid up front, not necessarily fully consumed experiment cost. Compare the same completed training/evaluation/export work, not equal rental hours or theoretical TFLOPS.

| Option | Rate used, USD/hour | Cost if billed exactly 3 hours | Maximum runtime versus selected $0.461/hour 5090 to break even |
|---|---:|---:|---:|
| Live Vast A4000 offer, detailed terms unaudited | 0.108 | 0.32 | 4.27× |
| Hyperstack A4000 + one public IP | 0.15672 | 0.47 | 2.94× |
| Live Vast 3090 offer, detailed terms unaudited | 0.193 | 0.58 | 2.39× |
| TensorDock 3090 catalog floor, before a sufficient-resource quote | 0.23 | 0.69 | 2.00× |
| AutoDL 4090 advertised | 0.280 | 0.84 | 1.65× |
| CloudRift 4090 advertised | 0.39 | 1.17 | 1.18× |
| Live Vast 4090 offer, detailed terms unaudited | 0.409 | 1.23 | 1.13× |
| AutoDL 5090 advertised | 0.414 | 1.24 | 1.11× |
| Featurize 5090 advertised | 0.447 | 1.34 | 1.03× |
| Selected Vast 5090 + 50GB disk | 0.461 | 1.38 | 1.00× |
| TensorDock 4090 catalog floor | 0.50 | 1.50 | 0.92× |
| CloudRift 5090 advertised, stock unresolved | 0.60 | 1.80 | 0.77× |
| Old Vast 5090 reference | 0.681 | 2.04 | 0.68× |

These are arithmetic scenarios, **not runtime predictions or all-in checkout quotes**. Hyperstack's documented root disk is included in the flavor; if another 100GB SSV is required, add about $0.00968/hour. TensorDock needs a resource quote above its default disk/RAM. Each threshold is `0.461 / alternative rate`; a value below 1 means that alternative must finish faster to break even. Transfer charges and additional storage are outside this simplified table.

Next step: verify the selected Vast rental's running state, SSH connection, actual hardware/software, and short warm-start throughput before treating the 2–3 hour planning range as a measured estimate. For this short continuation, moving providers solely to save well under a dollar may add more paid setup time than it saves. The selected cheaper host within the funded Vast account avoids another provider's initial funding requirement; it still requires setup and data transfer.
