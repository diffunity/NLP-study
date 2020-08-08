# NLG Decoding methods

### Greedy-decoding
$$w_t = \arg \max_{w} P(w|w_{1:t-1}) $$
at each time step $t$

### Beam-search
Consider total probabilites within a given window span (beam)
$$ \max \{ P(w_t|w_{1:t-1}) * P(w_{t+1}|w_{1:t}) *  \ldots P(w_{t+N}|w_{1:t+N-1}) \}$$


### Top-k sampling

Select top K number of tokens (Remove beyond k number of tokens)

<pre><code># from huggingface
indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
logits[indices_to_remove] = -float("inf")
</code></pre>

### Top-p sampling (nucleus sampling)
> [Holtzman et al.](https://arxiv.org/pdf/1904.09751.pdf)
> $$ \sum_{x \in V^(p)} P(x|x_{1:i-1}) \geq p $$
> where $V^{(p)} \subset V$ is the top-p sampled subset

Keep top p percentile for logits sorted in descending order

eg. [1, 2, 3, 4] -(softmax)-> [0.0321, 0.0871, 0.2369, 0.6439] -(cumsum)-> [0.0321, 0.1192, 0.3561, 1.0000]

if top_p = 0.3, keep = [True, True, False, False]

<pre><code># from hugging face
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > top_p
... # hugging face also keeps the first token above the threshold
indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
logits[indices_to_remove] = -float("inf")
</code></pre>

> they also said that using nucleus sampling makes the model "stochastic". Does this mean that the next token is taken completely at random from the sampled "nucleus"? 

### Temperatures
Diversifies the result of NLG 

$$ p(x= V_l | x_{1:i-1}) = \frac{\exp(u_l / t)}{\sum_{l'}{\exp (u'_l/t)}}$$


#### *Citation* 
* [huggingface blog](https://huggingface.co/blog/how-to-generate)
* [Nucleus Sampling - Holtzman et al.](https://arxiv.org/pdf/1904.09751.pdf)