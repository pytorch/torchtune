<h1>Reward modeling in torchtune RFC</h1>

<h2>Core issues</h2>

<ul>

  <li>We do not have an out-of-the-box toolkit to perform state-of-the-art reward modeling in torchtune;</li>

  <li>While the standard reward models are usually trained by common likelihood maximization across the preference dataset, there are several important observations related to the loss function;</li>

  <li>We do not support reward modeling and specific preference datasets (without prompt, only preference pairs). Furthermore, we do not support the LLM as judge framework, which is important to determine $\beta$ in the Bradley-Terry model while building a preference dataset; </li>

</ul>

<h2>Proposal</h2>

**Proposal:** Close these gaps using the tools that are already presented in torchtune.

<h3>Custom loss</h3>

We need a special loss for the reward modeling (it comes from the Bradley-Terry model):

$\sigma(r_1 - r_2)$

Where $r_1$ and $r_2$ are chosen and rejected rewards, respectively.

**How to implement in torchtune:** Basically just `-F.logsigmoid(rewards_chosen - rewards_rejected).mean()`. Probably one more file in the `rlhf` directory related to the reward modeling. It is important to make it flexible enough to make it possible to train with different objectives.

<h3>Reward centering</h3>

In many scenarios, it’s preferable to ensure that a reward model’s output is mean zero. This is often done by first calculating the model’s average score and then subtracting it.

https://arxiv.org/abs/2312.09244 introduces an auxiliary loss function designed to directly learn a centered reward model. This auxiliary loss minimizes the squared sum of the rewards, encouraging the model to naturally produce mean-zero outputs:

$(R(p, r_1) + R(p, r_2))^2$

This component is added to the main loss with some weighting coefficient $\omega$

**How to implement in torchtune:** basically just add this component to the loss: `centering_coefficient  torch.mean((rewards_chosen + rewards_rejected) * 2)`

<h3>Margin to the loss</h3>

It might be efficient to calculate margin from the rewards and add it to the BT loss (similarly to how it's done in llama papers); basically, it just requires an extra column in the dataset and a simple calculation.

**How to implement in torchtune:** We might need a custom reward modeling dataset in torchtune: with `margin` column and without a prompt. Then, just simple `-margin` from the rewards in the common loss.

<h3>Generation recipe utilization</h3>

This is the interesting one. We need to make it possible to do two things for the reward modeling:

<ol>

  <li>LLM as judge. <code>PreferenceTransform</code>? Given a dataset with prompt, response1, and response2. Transform it into: chosen, rejected, margin. Where margin is calculated through the $\beta$ of the responses. Furthermore, it is important to call the judge twice to remove contradictory pairs. I think this framework can be utilized for the online DPO either. </li>

  <li>API to generate lots of diverse responses with <b>different</b> (of different strength) models.</li>

  <li>same prompt -> cross prompt modeling? https://arxiv.org/pdf/2402.10958</li>

</ol>

<h3>Binary -> Embeddings representation of preferences</h3>

There is a way to infer multidimensional human preferences through some tricky PCA to identify orthogonal basis vectors, each

capturing a distinct human preference direction: https://arxiv.org/pdf/2502.13131

How to implement in torchtune: It will require a separate recipe, PCA, and separate dataset. 

<h1>General thoughts</h1>

Except for the last idea, we might eliminate the requirements of the separate recipe, but we need to create a new dataset type inheriting from PreferenceDataset (maybe some extra abstraction here?), loss, and transform.

Within this, the only thing that users might want to touch in configs to enable reward modeling is a loss section; basically, it might look like

```

loss:

  component: torchtune.modules.loss.RewardModelingLoss

  centering_rewards: True

  margins: True

```

We have strong evidence that cross-prompt modeling acts better than same-prompt, so I assume that we need to introduce it directly in torchtune, while some features might be delegated to users.

We might also want to introduce packing because of the possible size of the reward modeling datasets, but it is still not really trivial for the preference datasets.

