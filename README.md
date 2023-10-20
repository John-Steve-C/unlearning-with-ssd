# Toxic Comment Generation

I only use **pdr_tuning** in `forget_random_strategies.py`. The key algorithm lies in `ssd.py`

Train & test related files are `toxic_gen_train/test.py`, arguments are in `toxic_gen_train/test.sh`

Modified from [original code](https://github.com/if-loops/selective-synaptic-dampening/) to calculate the loss & accuracy in Huggingface's form:

- `metrics.py`
- `utils.py`
- `ssd.py`

I also write a model train in native Pytorch, called `train_in_pytorch.py`, you can have a try.

Remember to modify the **model_output_path** to your environment.

In my own PC, there is a *killed* output when the program running. I think the poor memory & GPU of my computer maybe the cause.

---

we delete the calculation of Mia & ZRF_score (all equals to zero)

I try the new accuracy calculation:

- decode the pred & label to text
- feed them to the `roberta_toxicity_classifier` to get the toxicity label (0/1)
- then calculate the accuracy