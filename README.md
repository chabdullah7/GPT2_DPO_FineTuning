#  GPT-2 DPO Fine-Tuning — Aligning AI with Human Preferences

### *Direct Preference Optimization (DPO) Training on GPT-2 using TRL and PEFT*

This project demonstrates **Direct Preference Optimization (DPO)** fine-tuning on **GPT-2** using **Hugging Face TRL**, **PEFT (LoRA)**, and **Datasets**.
It aligns the GPT-2 model with **human preference data**, enhancing response quality and reasoning compared to the baseline model.

---

##  Features

###  1. Direct Preference Optimization (DPO)

* Implements **DPO objective** to align model behavior with preferred outputs
* Uses a **reference model (GPT-2)** for stable optimization
* Integrates seamlessly with **TRL (Transformer Reinforcement Learning)**

###  2. Parameter-Efficient Fine-Tuning (LoRA)

* Uses **Low-Rank Adaptation (LoRA)** for memory-efficient training
* Fine-tunes selective layers (`c_attn`, `c_proj`)
* Reduces GPU memory cost while maintaining performance

###  3. Preference Dataset Integration

* Loads **UltraFeedback Binarized Dataset** from Hugging Face
* Preprocesses for `chosen` vs `rejected` text pairs
* Enables preference-based alignment in a controlled setup

###  4. Training Visualization

* Tracks both **training** and **evaluation loss** per epoch
* Generates loss curves for fine-tuning progress monitoring
* Logs intermediate checkpoints for experiment reproducibility

---

##  Core Architecture

| Component              | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| **Base Model**         | `GPT-2` (causal language model)                      |
| **Reference Model**    | Frozen GPT-2 for comparison                          |
| **Trainer**            | `DPOTrainer` from Hugging Face TRL                   |
| **Fine-Tuning Method** | LoRA (Low-Rank Adaptation)                           |
| **Dataset**            | `BarraHome/ultrafeedback_binarized`                  |
| **Libraries**          | Transformers, TRL, PEFT, Datasets, Torch, Matplotlib |
| **Environment**        | Google Colab / Local GPU                             |

---

### Model Responses

| Model            | Response                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------ |
| **GPT-2 (Base)** | "The answer is yes. The higher octane gasoline is more efficient and more fuel efficient." |
| **GPT-2 (DPO)**  | "The answer is yes. The higher octane gasoline is better for your car."                    |

---

##  Results Snapshot

| Metric                   | Description                          |
| ------------------------ | ------------------------------------ |
| **Base Model**           | GPT-2                                |
| **Fine-Tuned Objective** | DPO (Direct Preference Optimization) |
| **Dataset Size**         | 50 samples (subset of UltraFeedback) |
| **Optimizer**            | AdamW                                |
| **Learning Rate**        | 1e-4                                 |
| **Training Epochs**      | 5                                    |

---

##  Future Enhancements

* Scale training with **larger preference datasets**
* Add **reward model** evaluation
* Integrate **human feedback scoring**
* Use **GPT-2 Medium or GPT-J** for higher capacity alignment
* Deploy as an **interactive comparison app** (Streamlit/Gradio)

---

##  Author

**Abdullah**

* GitHub: [github.com/chabdullah7](https://github.com/chabdullah7)
* LinkedIn: [linkedin.com/in/ch-abdullah-b537951a](https://www.linkedin.com/in/ch-abdullah-b537951a)
