## Single Head Attention

note  $L=$  sub-sequence length
$$
\begin{align*}
	X &\in \R^{L\times d_{model}} \\
	W_K \in \R^{d_{model}\times d_{key}},
	W_V &\in \R^{d_{model}\times d_{value}},
	W_Q \in \R^{d_{model}\times d_{key}}\\
	
	K&= XW_K \\
	&=(L , d_{model})(d_{model} , d_{key})\\
	&= (L,d_{key})\\
	
	V&= XW_V \\
	&=(L , d_{model})(d_{model} , d_{value})\\
	&= (L,d_{value})\\
	
	Q&= XW_Q \\
	&=(L , d_{model})(d_{model} , d_{key})\\
	&= (L,d_{key})\\
	
	A_{dot} &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_{model}}}\right)V\\
	&=(L,d_{key})(L,d_{key})^T(L,d_{value})\\
	&=(L,d_{key})(d_{key},L)(L,d_{value})\\
	&=(L,d_{value})\\
	
	O &= A_{dot}W_O\\
	&= (L,d_{value})(d_{value},d_{model})\\
	&= (L,d_{model})
\end{align*}
$$

## Multi-Head Attention

concat and project the heads so its shape comes back to $d_{value}$ , the same as a single head attention
$$
\begin{align*}
W_{MHA}&\in \R^{H\cdot d_{value}\times d_{value}}\\
MHA &= [A_{dot}^1,A_{dot}^2,\cdots,A_{dot}^H]{W_{MHA}}\\
&= (L,H\cdot d_{value})(H\cdot d_{value},d_{value})\\
&=(L,d_{value})
\end{align*}
$$

## Memory Attention

note $s$ as the state, it starts from index $0$ of the whole sequence, and steps ahead by $L$​ 

#### Memory Retrieval

note $\sigma$ as nonlinear activation function 
$$
\begin{align*}
A_{mem}\in \mathbb{R}^{L\times d_{value}}, M_s \in &\R^{d_{key}\times d_{value}}, Q\in \R^{L \times d_{key}},z_s \in \R^{d_{key}}\\
A_{mem} &= \frac{\sigma(Q)M_{s-1}}{\sigma(Q)z_{s-1}}
\end{align*}
$$

#### Memory Update

$$M_s \leftarrow M_{s-1} + \sigma (K)^TV \\ z_s \leftarrow z_{s-1} + \sum_{t=1}^N\sigma(K_t)$$

*incremental update*

$$\begin{align*}M_s \leftarrow M_{s-1} + \sigma (K)^T\left(V -\frac{\sigma(K)M_{s-1}}{\sigma(K)z_{s-1}}\right)\end{align*}$$

#### Long-term context injection

$$A=sigmoid(\beta)\odot A_{mem}+(1-sigmoid(\beta))\odot A_{dot}$$