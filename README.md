GPT-2 Model Implementation Report
Introduction
In this report, I document the implementation of the GPT-2 model using Python and PyTorch. The goal was to create a GPT-2 model with 125 million parameters, focusing on key aspects such as multi-head self-attention, feed-forward networks, and positional encoding. The implementation follows the architecture described in the GPT-2 paper's Sections 1 and 2. Additionally, I referred to Andrej Karpathy’s nanogpt repository and the makemore series for guidance.

Implementation Details
GPT-2 Model Architecture
Token Embeddings: Utilized nn.Embedding in PyTorch to create token embeddings for vocabulary tokens.
Positional Embeddings: Employed nn.Embedding for positional embeddings and added them to token embeddings to incorporate positional information.
Transformer Encoder: Developed the Transformer Encoder using stacked EncoderLayers with multi-head self-attention and feed-forward networks.
Encoder Layer: Implemented each EncoderLayer consisting of multi-head self-attention and feed-forward networks, followed by layer normalization.
Multi-head Self-Attention: Developed a separate module for multi-head self-attention following the scaled dot-product attention mechanism.
Testing and Validation
Model Instantiation: Created an instance of the GPT-2 model with a specified vocabulary size.
Checkpoint Loading: Loaded original GPT-2 125M model checkpoints using PyTorch's load_state_dict() function.
Sample Prediction: Generated sample predictions using the loaded checkpoints to verify the correctness of the model.
Challenges Encountered
Challenges:
Understanding Positional Encoding: Initially faced challenges in grasping the concept of positional encoding and incorporating it into the model.
Handling Checkpoint Loading: Encountered issues while loading the original GPT-2 checkpoints due to mismatches in model architecture and checkpoint keys.
Solutions:
Positional Encoding: Studied the original GPT-2 paper's descriptions and external resources to gain a deeper understanding of positional encoding and successfully implemented it in the model.
Checkpoint Loading: Carefully matched the keys of the loaded checkpoints to the corresponding model parameters and resolved discrepancies in model architecture for successful loading.
Results and Validation
Model Functionality: The implemented GPT-2 model successfully generated predictions and exhibited functionality similar to the original GPT-2 architecture.
Sample Predictions: Sample predictions showed coherent outputs, indicating that the model was learning and predicting sequences effectively.
Conclusion
The implementation of the GPT-2 model encompassed key components such as multi-head self-attention, feed-forward networks, and positional encoding. By following architectural descriptions, referring to relevant resources, and overcoming encountered challenges, a functional GPT-2 model with 125 million parameters was created in PyTorch.

References
GPT-2 Paper: Link to the GPT-2 Paper
nanogpt Repository: Link to Andrej Karpathy’s nanogpt repository
makemore Series: Link to the makemore series


1. Rotary Positional Embedding
Implementation: Replace the original positional embeddings in the GPT-2 model with Rotary embeddings as described in Su et al.'s RoFormer.
Steps:
Understand the concept of Rotary embeddings and how they differ from traditional positional encodings.
Modify the positional encoding module in your GPT-2 implementation to incorporate Rotary embeddings.
Update the model architecture to utilize Rotary positional embeddings instead of the original method.
Assessment:
Evaluate the impact on model performance, computational efficiency, and memory usage.
Comment on any observed improvements or drawbacks in terms of language generation quality, convergence speed, or any other metrics of interest.
2. Group Query Attention
Implementation: Integrate the Group Query Attention mechanism into your GPT-2 model based on insights from Ainslie et al.'s GQA paper.
Steps:
Study the GQA mechanism and how it modifies the standard attention mechanism in Transformers.
Modify the attention mechanism in your EncoderLayer to incorporate Group Query Attention.
Assess how the modified attention mechanism affects the model's behavior and performance.
Assessment:
Analyze the impact of Group Query Attention on the model's ability to capture long-range dependencies, computational efficiency, or any other relevant aspects.
Compare the model's performance before and after incorporating Group Query Attention.
3. Sliding Window Attention
Implementation: Implement the Sliding Window Attention mechanism into your GPT-2 model, inspired by Beltagy et al.'s Longformer.
Steps:
Understand the concept of Sliding Window Attention and its advantages in handling long sequences.
Modify the attention mechanism in your EncoderLayer to incorporate Sliding Window Attention.
Assess the effects of this attention mechanism on model performance and capabilities.
Assessment:
Evaluate the model's efficiency in processing longer sequences with Sliding Window Attention.
Comment on any trade-offs, improvements, or challenges encountered while implementing Sliding Window Attention.
Deliverable and Evaluation
Deliverable: Provide Python code showcasing any one, two, or all three modifications in the GPT-2 model, along with relevant comments and documentation explaining the changes.
Evaluation Scheme: Each successful implementation of the mentioned changes will be assessed based on the specific criteria mentioned for each feature, i.e., Rotary Positional Embedding (15 points), Group Query Attention (10 points), and Sliding Window Attention (15 points).
Reporting
Document the process of incorporating each modification, including references to relevant papers or resources.
Discuss the impact of each change on the model's size, capabilities, potential pitfalls, and any observed improvements or challenges.
Present results, comparisons, or metrics to support the assessment of the model after each modification.
Ensure to thoroughly test and validate the model after incorporating each architectural change to understand its impact on the GPT-2 model's behavior and performance. Detailed reporting and assessment are crucial to evaluate the effectiveness of these modifications in enhancing the model's capabilities.

Creating a training loop compatible with single GPU, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP) setups involves adapting the codebase to handle parallel training across multiple GPUs and implementing sharding of model parameters, gradients, and optimizer state. Below is a guideline on how to accomplish this task:

Single GPU Training Loop
Initialization: Set up the model, optimizer, loss function, and data loaders for training on a single GPU.
Training Loop: Implement a loop to iterate through batches of data, perform forward and backward passes, update model parameters using the optimizer, and calculate the training loss.
Distributed Data Parallel (DDP)
Setup DDP: Modify the training loop to incorporate PyTorch's torch.nn.parallel.DistributedDataParallel module.
Initialize Process Group: Initialize the process group, set up distributed training environment, and configure rank and world size for each GPU.
Wrap Model with DDP: Wrap the model with DistributedDataParallel to parallelize computations across multiple GPUs.
Adapt Training Loop: Adjust the training loop to handle DDP-related operations, such as broadcasting gradients, synchronizing gradients across GPUs, and averaging losses.
Fully Sharded Data Parallel (FSDP)
Understanding FSDP: Familiarize yourself with Fully Sharded Data Parallel (FSDP) as described in the Gupta et al. paper.
Shard Model and Optimizer: Modify the model and optimizer to shard parameters using FSDP.
Implement FSDP in Training Loop: Update the training loop to handle FSDP-related operations, such as sharding gradients and optimizer state, and ensuring synchronization between shards.
Handle Gradient Reduction: Adapt the training loop to perform gradient reduction and aggregation across shards.
Deliverable and Evaluation
Deliverable: Provide a Python script showcasing the functional training loop compatible with single GPU, DDP, and FSDP settings. Include comprehensive documentation explaining the adaptations made for each setting.
Evaluation Scheme: Each implemented feature – Single GPU (10 points), DDP (10 points), and FSDP (20 points) – will be evaluated based on the successful execution, compatibility, and proper handling of parallel training across GPUs.
