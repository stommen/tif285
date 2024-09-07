# Problem set 1 notes

## Code notes

array.reshape(-1,1): row vector --> column vector

ADAM: One momentum part (m, like a ball rolling downwards) and one squared gradient-part

## Comparison of Gradient Descent Algorithms

### 1. Batch Gradient Descent (BGD)

#### Description

- Uses the entire training data to calculate the gradient of the cost function at each update.
- The optimizer updates the model's parameters after calculating the average gradient over the entire data.

##### Advantages

- Stable and accurate gradient estimation.
- Often converges to the global minimum if an appropriate learning rate is used.

##### Disadvantages

- Can be very slow with large datasets as the entire dataset must be processed for each parameter update.
- Can get stuck in local minima if the data is complex.

#### Choice of Learning Rate

- If the learning rate is too high, the algorithm can overshoot the minimum points and become unstable.
- If the learning rate is too low, convergence can be very slow.

### 2. Stochastic Gradient Descent (SGD)

<!-- markdownlint-disable MD024 -->
#### Description

- Updates the model's parameters based on a single training data point at a time.
- The gradient is calculated and parameters are updated for each individual training example.

##### Advantages

- Faster than BGD for large datasets as it does not require processing the entire dataset at once.
- Introduces randomness which can help avoid local minima and lead to better generalization.

##### Disadvantages

- Gradient estimation is noisier and can lead to oscillations around the optimal solution.
- May require more epochs to converge compared to BGD.

#### Choice of Learning Rate

- Must be carefully chosen to balance between fast convergence and stability.
- Dynamic adjustment of the learning rate during training (e.g., decreasing learning rate) can help optimize performance.

### 3. Mini-Batch Gradient Descent (MBGD)

#### Description

- A mix of BGD and SGD; updates parameters with a small, random subset of the training data (mini-batch).
- Typically, a mini-batch size between 10 and 1000 samples is used.

##### Advantages

- Balances between speed and precision, offering a compromise between BGD stability and SGD speed.
- Can yield better results and faster convergence compared to BGD and SGD when the data is large.

##### Disadvantages

- Can be more complex to implement and fine-tune, especially regarding the choice of batch size and learning rate.

#### Choice of Learning Rate

- Similar to SGD, but also affected by the batch size. A good strategy is to start with a reasonable learning rate and adjust based on performance and convergence speed.
