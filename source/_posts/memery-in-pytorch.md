---
title: 在PyTorch中可视化和理解GPU内存
mathjax: true
date: 2026/1/13 20:46:25
img: https://d00.paixin.com/thumbs/1772227/30081205/staff_1024.jpg
excerpt: 教你不再CUDA:OUT OF MEMORY
---
```bash
RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 7.93 GiB total capacity; 6.00 GiB already allocated; 14.88 MiB free; 6.00 GiB reserved in total by PyTorch)
```

虽然很容易看到GPU内存已经满了，但理解为什么以及如何修复它可能更具挑战性。在本教程中，我们将逐步介绍如何在训练过程中可视化和理解PyTorch中的GPU内存使用情况。我们还将了解如何估计内存需求和优化GPU内存使用。

PyTorch提供了一个方便的工具来可视化GPU内存使用：

```python
import torch
from torch import nn

# Start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = nn.Linear(10_000, 50_000, device ="cuda")
for _ in range(3):
    inputs = torch.randn(5_000, 10_000, device="cuda")
    outputs = model(inputs)

# Dump memory snapshot history to a file and stop recording
torch.cuda.memory._dump_snapshot("profile.pkl")
torch.cuda.memory._record_memory_history(enabled=None)
```

运行这段代码会生成一个`profile.pkl`文件，其中包含了执行过程中GPU内存使用的历史记录。您可以在https://pytorch.org/memory_viz上可视化此历史。

通过拖放`profile.pkl`文件，你会看到这样的图形：

![](https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/simple_profile.png)

让我们把这个图分解成几个关键部分：

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/simple_profile_partitioned.png)

1. 模型创建：内存增加2 GB，对应于模型的大小：

   $10000 \times 50000$ weight $+ 50000$ biases in $float32(4\ bytes)$=$5 \times 10^8 \times 4 bytes = 2GB$​

   这个内存（蓝色）在整个执行过程中都保持不变。

2. 创建输入张量（第一个循环）：匹配输入张量大小，内存增加200 MB：

   $5000 \times 10000$ elements in $float32(4\ bytes)=0.2GB$

3. 前向传递（第1个循环）：输出张量的内存增加1 GB：

   $5000\times50000$ elements in $float32(4\ bytes)=(25\times10^7)\times4\ bytes=1GB$​ 

4. 输入张量创建（第2个循环）：对于一个新的输入张量，内存增加200 MB。在这一点上，您可能希望从步骤2的输入张量得到释放。然而，它不是：模型保留了它的激活，因此即使张量不再分配给变量`inputs`，它仍然被模型的前向传递计算引用。模型保留了它的激活，因为这些张量是神经网络中的反向传播过程所需要的。试试`torch.no_grad()`；看看区别。

5. 前向传递（第二个循环）：新输出张量的内存增加了1 GB，按步骤3计算。

6. 释放第一个循环激活：在第二个循环向前传递之后，可以释放第一个循环（步骤2）的输入张量。模型的激活保存第一个输入张量，被第二个循环的输入覆盖。一旦第二个循环完成，第一个张量不再被引用，它的内存可以被释放。

   第二次 forward 完成

   计算图更新

   第一次 forward 的输入激活不再被引用

7. 更新`output`；：将步骤3的输出张量重新分配给变量`output`；前一个张量不再被引用并被删除，释放其内存。

8. 输入张量创建（第三个循环）：与步骤4相同。

9. 前向传递（第三循环）：和第5步一样。

10. 释放第二格循环激活：释放步骤4的输入张量。

11. 再次更新`output`：从步骤5输出的张量被重新分配给变量`output`，释放之前的张量。

12. 代码执行结束：释放所有内存。

前面的例子是简化的。在真实场景中，我们经常训练复杂的模型，而不是单个线性层。此外，前面的例子不包括训练过程。在这里，我们将研究GPU内存在一个真正的大型语言模型（LLM）的完整训练循环期间的行为。

```python
import torch
from transformers import AutoModelForCausalLM

# Start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for _ in range(3):
    inputs = torch.randint(0, 100, (16, 256), device="cuda")  # Dummy input
    loss = torch.mean(model(inputs).logits)  # Dummy loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Dump memory snapshot history to a file and stop recording
torch.cuda.memory._dump_snapshot("profile.pkl")
torch.cuda.memory._record_memory_history(enabled=None)
```

提示：在分析时，限制步骤的数量。每个GPU内存事件都会被记录，文件可能会变得非常大。例如，上面的代码生成一个8 MB的文件。

下面是这个例子的内存配置：

![](https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/raw_training_profile.png)

这个图比前面的例子更复杂，但我们仍然可以一步一步地分解它。注意这三个尖峰，每个尖峰都对应于训练循环的一次迭代。让我们简化图形以使其更容易解释：

![](https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/colorized_training_profile.png)

1. 模型初始化(`model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")`)

第一步是将模型加载到GPU上。模型参数（蓝色）占用内存，直到训练结束。

2. 前向传播(`model(inputs)`)

在正向传递期间，激活（每层的中间输出）被计算并存储在内存中以进行反向传播。这些激活，用橙色表示，一层一层地增长，直到最后一层。损失在橙色区域的峰值计算。

3. 反向传播(`loss.backward()`)

梯度（黄色）在此阶段计算并存储。同时，由于不再需要，激活被丢弃，导致橙色区域缩小。黄色区域表示梯度计算的内存使用情况。

4. 优化器更新(`optimizer.step()`)

梯度用于更新模型参数。最初，优化器本身被初始化（绿色区域）。这个初始化只执行一次。之后，优化器使用梯度来更新模型参数。为了更新参数，优化器临时存储中间值（红色区域）。更新之后，梯度（黄色）和中间优化器值（红色）都被丢弃，从而释放内存。

至此，一次训练迭代完成。该过程在剩余的迭代中重复，产生图中可见的三个内存峰值。

这样的训练配置文件通常遵循一致的模式，这使它们有助于估计给定模型和训练循环的GPU内存需求。

从上面的部分来看，估计GPU内存需求似乎很简单。所需的总内存应该对应于内存配置文件中的最高峰值，该峰值发生在正向传递期间。在这种情况下，内存需求是（蓝绿橙）： $Model Parameters+Optimizer State+Activations$

就这么简单吗？实际上，有一个陷阱。根据训练设置的不同，配置文件可能看起来不同。例如，将批量大小从16减少到2会改变图片：

```python
- inputs = torch.randint(0, 100, (16, 256), device="cuda")  # Dummy input
+ inputs = torch.randint(0, 100, (2, 256), device="cuda")  # Dummy input
```

![](https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/colorized_training_profile_2.png)

现在，最高峰值出现在优化过程中，而不是前向过程中。在这种情况下，内存需求变为（蓝绿黄红）：$Model Parameters+Optimizer State+Gradients+Optimizer Intermediates$

为了推广内存估计，我们需要考虑所有可能的峰值，而不管它们是发生在正向传递还是在优化器步骤中。

- 模型参数是最容易估计的。

$Model Parameters+Optimizer State+max(Gradients+Optimizer Intermediates,Activations)$

现在我们有了方程，让我们看看如何估计每个分量。

$Model Memory=N×P$

其中$N$是参数数量，$P$是精度（字节为单位，例如`float32`就是4）

例如，一个有15亿个参数且精度为4字节的模型需要：

$Model Memory=1.5×10^9×4bytes=6GB$

- 优化器状态所需的内存取决于优化器类型和模型参数。例如，`AdwmW`；优化器为每个参数存储两个矩（第一和第二）。这使得优化器的状态大小：

$Optimizer State Size=2×N×P$

- 激活所需的内存很难估计，因为它包括前向传递期间计算的所有中间值。为了计算激活内存，我们可以使用一个前向hook来测量输出的大小：

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")

activation_sizes = []

def forward_hook(module, input, output):
    """
    Hook to calculate activation size for each module.
    """
    if isinstance(output, torch.Tensor):
        activation_sizes.append(output.numel() * output.element_size())
    elif isinstance(output, (tuple, list)):
        for tensor in output:
            if isinstance(tensor, torch.Tensor):
                activation_sizes.append(tensor.numel() * tensor.element_size())

# Register hooks for each submodule
hooks = []
for submodule in model.modules():
    hooks.append(submodule.register_forward_hook(forward_hook))

# Perform a forward pass with a dummy input
dummy_input = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
model.eval()  # No gradients needed for memory measurement
with torch.no_grad():
    model(dummy_input)

# Clean up hooks
for hook in hooks:
    hook.remove()

print(sum(activation_sizes))  # Output: 5065216
```

对于Qwen2.5-1.5B模型，这为每个输入token提供了5,065,216个激活。要估计输入张量的总激活内存，使用：$ Activation Memory=A×B×L×P$

其中A是每个token的激活数量，B是批量大小，L是序列长度

然而，直接使用这种方法并不总是实用的。理想情况下，我们希望在不运行模型的情况下用启发式方法估计激活内存。此外，我们可以直观地看到，更大的模型具有更多的激活。这就引出了一个问题：模型参数的数量和激活的数量之间是否存在联系？

不是直接的，因为每个令牌的激活数量取决于模型架构。然而，llm往往具有相似的结构。通过分析不同的模型，我们观察到参数数量和激活数量之间大致的线性关系：

![](https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/activation_memory_with_global_regression.png)

这种线性关系允许我们使用启发式方法来估计激活值：

$A=4.6894×10^{−4}×N+1.8494×10^6$

虽然这只是一个近似值，但它提供了一种实用的方法来估计激活内存，而不需要对每个模型进行复杂的计算。

梯度更容易估计。梯度所需的内存与模型参数相同：

$Gradients Memory=N×P$

更新模型参数时，优化器存储中间值。这些值所需的内存与模型参数相同：

$Optimizer Intermediates Memory=N×P$

总而言之，训练模型所需的总内存为：

$Total Memory=Model Memory+Optimizer State+max(Gradients,Optimizer Intermediates,Activations)$

通过以下组件：

- **Model Memory**: $N×P$
- **Optimizer State**: $2×N×P$
- **Gradients**: $N×P$
- **Optimizer Intermediates**: $N×P$
- **Activations**:$ A×B×L×P, A=4.6894×10^{−4}×N+1.8494×10^6$

