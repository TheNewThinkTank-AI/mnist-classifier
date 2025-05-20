# Deep Neural Networks

Implementation details for our dense DNN model

## Why 128 Neurons?

For each of the hidden layers we have chosen 128 neurons.
The choice of 128 here is somewhat arbitrary — it's a hyperparameter — but it follows some common practices:

- Power of 2: 128 is a power of 2 (like 32, 64, 256), which can help with memory alignment and GPU optimization.
- Balanced Complexity: It's large enough to capture meaningful patterns in the data (like edges or shapes in MNIST), but not so big that it causes overfitting or excessive computation.
- Empirical Performance: Through trial and error, people have found that 128 often works well for small datasets like MNIST.
- Historical Precedent: Many tutorials and papers use 128 as a default starting point for hidden units.

---

## Model Compilation

### Why Adam optimizer is a Good Choice

Adam stands for Adaptive Moment Estimation , and it's one of the most popular optimization algorithms in deep learning.

#### Key Advantages

- Adaptive Learning Rates:
Each weight gets its own learning rate that adapts during training — faster convergence.
- Combines Momentum & RMSProp: Uses both momentum (to accelerate SGD) and adaptive scaling (to handle noisy gradients).
- Robust to Hyperparameters: Works well with default settings (like learning rate = 0.001), so less tuning needed.
- Good Performance on MNIST: For simple datasets like MNIST, Adam usually converges quickly and reliably.

#### How It Works (Simplified)

Keeps track of moving averages of gradients (first moment) and gradient squared (second moment). Adjusts each parameter update based on these statistics. Helps avoid issues like vanishing gradients and oscillations during training. In short: Adam is fast, stable, and works well out-of-the-box, especially for small networks and datasets like MNIST.

---

### sparse_categorical_crossentropy

#### Why This Loss Function?

This loss is specifically designed for multi-class classification problems where:

The labels (y_train, y_test) are integers (e.g., 0 through 9)
The output layer uses softmax activation to produce a probability distribution over classes.

#### What Does It Do?

It compares the predicted probability distribution (from softmax) with the true label (e.g., class 3), and penalizes predictions that are far from the true value.

#### Example

If your model predicts:

```python
[0.05, 0.02, 0.03, 0.8, ...]  # Predicting class 3
```

But the true label is `3`

The loss will be low because the model assigned high probability to the correct class.

But if the model says:

```python
[0.4, 0.3, 0.2, 0.1, ...]  # Not confident about class 3
```

Then the loss will be higher.

---

### Why Track Accuracy?

Accuracy measures how often the model makes the correct prediction.
It's easy to understand: e.g., "97% accuracy" means 97% of predictions were correct.

#### When Accuracy Might Be Misleading

On imbalanced datasets (e.g., 90% of samples are class 0), accuracy can be misleading. But for MNIST, classes are balanced, so accuracy is a valid and useful metric.
You can also add more metrics (like precision, recall, F1-score) if you want deeper insight into performance per class.

---

## Model Fitting

### Why Use 3 Epochs?

Let's first define what an epoch is:
An epoch is one full pass through the entire training dataset.

#### Why 3 Might Be Used

- Speed: Training for only 3 epochs is fast — useful for quick experimentation or testing code.
- Avoid Overfitting: If the dataset is very large or complex, fewer epochs may help prevent overfitting (not really the case for MNIST).
- Baseline Start: Often people start with a small number of epochs to see if the model learns at all before increasing.

#### Is 3 Enough?

For MNIST, which is a simple and clean dataset, even 1 epoch might give decent results (~90%+ accuracy). However:<br><br>

1    epoch  -> ~90-92% approximate accuracy<br>
3    epochs -> ~95-97%<br>
5-10 epochs -> ~98%+<br><br>

So while 3 epochs is better than 1, it's still on the lower side for achieving the best possible performance. Usually, people train MNIST models for 5-10 epochs to reach near-optimal accuracy.
