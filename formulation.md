# Assumption

对一个孤立系统而言，系统的真实状态一定可以写作：
$$
z=(z_1​,…,z_k​)∈Z⊆R^k
$$
且系统一定存在真实动力学：
$$
\dot z=F(z),
$$
如果将时钟设置得充分小，那么这等价于：
$$
z_{t+1} = z_t + \delta F(z_t)
$$


这里 $z$ 可以包含位置、速度、半径、质量等物理参数；$F$ 是真实动力学。视频帧不是状态本身，而是状态经过渲染函数 $R$ 得到的图像：
$$
I_t=R(z_t),\qquad V=(I_1,\ldots,I_L).
$$
举个例子：若对于一个在真空中自由下落的小球与重力场所构成的系统而言，系统的状态可以被序对 （小球质量，小球速度，当前高度）描述，动力学则是 $ F((m, v, h)) = (0, 9.81, -v, 0)$

值得一提的是，在孤立力学系统中，”状态“就是物体的基本属性（如质量、摩擦因数）和物体的零阶、一阶几何量（位置、速度），加速度上隐含在了动力学方程 $F$ 中。这里以及下文我都使用力学系统做例子，是因为我认为力学系统已经足够涵盖世界模型的大部分使用场景了。

另一个有意思的点是，$R$ 作为确定性函数（理论上），它应当不是单射的，因为可能不同真实状态的系统的快照看起来一模一样。

特别注意：以上 $z$ 与 $F$ 都是客观存在的 target，而非模型学得的结构。



# ID & OOD 

最大的难点莫过于像传统统计学习那样定义 domain，这里我尝试过诸多角度，我认为最契合直觉、最可能对我们的理论有帮助的一种定义如下。

假定数据的构造过程是：先采样系统的 $z_0 \sim P_0$，这个初始分布可任意指定，然后使用固定的某真实动力学 $F$ rollout 出轨迹 $\{z_t\}_{t = 0}^{N}$，最后能得到一个状态分布 $P_\text{train}(z)$ 和一个渲染得到的训练集 $D_\text{train}$；在测试时采用另一分布采样 $z_0$ 并使用同样的 $F$ 来 rollout 轨迹并得到 $P_\text{test}(z)$ 与 $D_\text{test}$。从理论上讲，我们只需要通过控制 $z_0$ 的分布是否相同即可分别构造 ID 与 OOD 的测试集。

也就是说，以上我们关注的是 OOD 问题下的一个子问题，**我把它们称为 state OOD**，分布外的测试集与训练集应当共享同一系统动力学，只是系统真实状态的分布不同，或者更极端一些，测试集的真实状态没见过。

然而我们很难不面对更棘手的情况，那就是碰到我们从未见到过的物理规律。比如两小球相对运动，这一过程是牛二定律在作用于系统的状态机上；相碰的瞬间，系统状态发生突变，这一瞬间是动量守恒+弹性系数方程驱使系统改变状态（请允许我这里不把他们认定为牛二律的一种极限形式）。这种连环境的真实动力学 $F$ 都不再一致时的测试样本，**我把它们称为 law OOD**。

一般来说 state OOD 可以通过对动力学网络施加很强的归纳偏置来解决， law OOD 则一定不可能在没见过相关示例的情况下（无论采用怎样的架构）自主涌现：好比如果训练时只看到两小球相对运动，那么模型也许可以做到续写两小球相对运动的动画，但一定没有办法生成出两小球碰撞后各自反向的画面。

# Motivation

现有的 VWM 架构大多是利用 diffusion 强大的生图能力搭配 scaling 来端到端地生成下一帧画面，已有工作指出（[How Far is Video Generation from World Model: A Physical Law Perspective](https://arxiv.org/pdf/2411.02385)），这类架构并不具备学到物理定律结构的能力，进而不具备 OOD 泛化能力——这是因为它把颜色、背景等预测无关状态同位置、速度、质量等预测有关状态杂糅在一起，而且从结果上看，模型并不是在物理因果变量上推理，而更像是在对视觉属性做 case-based retrieval

当然，我认为有足够的理由相信，继续 scale up 多种类的样本，扩散类的世界模型是可以涌现出对物理规律的理解进而实现 state OOD 泛化，并且足够数量和种类的样本也的确足够涵盖自然界所有（大概）可能遇到的动力学，故而没有 law OOD 泛化的需要。这种几乎没有偏置的黑盒叠加上海量数据驱动的方法的确保有极高的上限，但训练代价极高，且研究价值不大。

我们因此关注的是，怎样设计一个能真正学到一定物理规律的架构。目前已有很多工作通过给前向网络加各色各样的归纳偏置来增强模型学到物理结构的能力，比如HNN、LNN等，可这些成果仍然有其局限性：它们都只适用于一种或一类环境动力学，难以做到 law OOD 泛化。于是我们提出了 CLaw-WM: Continual Law-discovery World Model，这种架构可以持续学到新的动力学规律，并且会在学习过程中不断巩固之前学到的规律。



# Formulation

设连续到来的数据流为：
$$
\mathcal{D}_1,\mathcal{D}_2,\dots,\mathcal{D}_n
$$
每个时刻构造上下文窗口（否则不足以支持对 $z_t$ 的推断）：
$$
C_t=(I_{t-K+1},\dots,I_t)
$$
用表征映射器估计当前真实状态：
$$
\hat z_t=E_\theta(C_t)
$$
维护一个动力学头库：
$$
\mathcal{F}_n=\{F_\phi^1,\dots,F_\phi^{M_n}\}
$$
每一帧或每隔 $m$ 帧选择一次动力学头（选头规则需要摸索）：
$$
i_t=\pi(C_t,\mathcal{F}_n)
$$
由被选中的动力学头预测下一状态：
$$
\hat z_{t+1}=F_\phi^{i_t}(\hat z_t)
$$
再由渲染器生成下一帧：
$$
\hat I_{t+1}=R_\psi(\hat z_{t+1})
$$
合并为：
$$
\boxed{
\hat I_{t+1}
=
R_\psi
\left(
F_\phi^{\pi(C_t,\mathcal{F}_n)}
\left(
E_\theta(C_t)
\right)
\right)
}
$$
训练时，对新数据评估已有动力学头（这里的选头机制最好和上面一样，anyway也需要摸索）：
$$
\mathcal{E}_i(\mathcal{D}_n)
=
\sum_{(C_t,I_{t+1})\in \mathcal{D}_n}
d
\left(
R_\psi(F_\phi^i(E_\theta(C_t))),
I_{t+1}
\right)
$$
若：
$$
\min_i \mathcal{E}_i \leq \tau
$$
则复用最优旧头：
$$
i^\star=\arg\min_i \mathcal{E}_i
$$
并通过 few-shot / fine-tuning / OWL 更新：
$$
\theta,\phi_{i^\star},\psi
$$
若：
$$
\min_i \mathcal{E}_i > \tau
$$
则扩展新头：
$$
\mathcal{F}_{n+1}
=
\mathcal{F}_n
\cup
\{F_\phi^{M_n+1}\}
$$
并训练：
$$
\theta,\phi_{M_n+1},\psi
$$
同时用持续学习正则巩固旧头：
$$
\mathcal{L}_{total}
=
\mathcal{L}_{new}
+
\lambda_{CL}\mathcal{L}_{CL}
$$
其中：
$$
\mathcal{L}_{new}
=
d(\hat I_{t+1},I_{t+1})
$$
可由 replay、regularization、distillation 或 OWL 给出。

数据流向：
$$
\boxed{
C_t
\xrightarrow{E_\theta}
\hat z_t
\xrightarrow{\pi}
F_\phi^{i_t}
\xrightarrow{}
\hat z_{t+1}
\xrightarrow{R_\psi}
\hat I_{t+1}
}
$$
并且：
$$
\boxed{
\text{每帧动态选头；旧头可复用补齐；无合适头则扩展新头；持续学习巩固旧规律。}
}
$$
![ChatGPT Image 2026年6月1日 18_23_23](C:\Users\Lenovo\Downloads\ChatGPT Image 2026年6月1日 18_23_23.png)



