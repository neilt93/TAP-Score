# SOTA Identification (Requirement 1)

## Primary Baseline: Diffusion Policy

**Paper:** Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023, IJRR 2024)

**Why it's SOTA:**
- State-of-the-art on multiple manipulation benchmarks (Push-T, RoboMimic, etc.)
- Handles multimodal action distributions via diffusion
- Stable training, strong generalization within distribution
- Well-documented codebase with reproducible results

**What it does:**
- Takes observation history as input
- Outputs action chunk (sequence of actions) via denoising diffusion
- Uses receding-horizon control (predict chunk, execute first few actions, replan)

**Repo:** https://github.com/real-stanford/diffusion_policy

---

## Alternative Methods (for context)

| Method | Key Idea | Comparison to Diffusion Policy |
|--------|----------|-------------------------------|
| **IBC** (Implicit BC) | Energy-based model for actions | Less stable training |
| **Behavior Transformer** | Autoregressive action prediction | Similar performance, different architecture |
| **LSTM-GMM** | Mixture density network | Struggles with multimodality |
| **ACT** | Action chunking transformer | Comparable, different chunk handling |

---

## Gap Addressed by This Project

**What Diffusion Policy lacks:** Built-in OOD detection. When observations shift (noise, occlusion, brightness), the policy continues generating actions without knowing they may be wrong.

**Our contribution:** TAP-Score provides an "actionness" signal that detects when proposed actions are off the expert manifold, enabling:
1. OOD detection at runtime
2. Optional intervention via resampling

---

## References

1. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
2. Florence et al., "Implicit Behavioral Cloning", CoRL 2021
3. Shafiullah et al., "Behavior Transformers", NeurIPS 2022
4. Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), RSS 2023
