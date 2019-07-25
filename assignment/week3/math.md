已知渡河总时长不变，则有

${{S_1} \over{V * \cos \alpha_1}}+ {{S_2} \over{V * \cos \alpha_2}} + ... + {{S_n} \over{V * \cos \alpha_n}}  = T$

要求 $dH$ 如下式的极小值



$dH ={{S_1} \over{V * \cos \alpha_1}}*(V*sin\alpha_1 +V_1) + {{S_2} \over{V * \cos \alpha_2}} *(V*sin\alpha_2 +V_2) + ... + {{S_n} \over{V * \cos \alpha_n}} *(V*sin\alpha_n +V_n) $

可看做是如下的优化问题

$min_x f(x)$

*s*.*t*. *g*(***x***)=0

![f与g的等高线图](https://img-blog.csdn.net/20171030202204122?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDc5MjMwNA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

令$T(n) ={{S_1} \over{V_1 * \cos \alpha_1}} + {{S_2} \over{V_2 * \cos \alpha_2}} + ... + {{S_n} \over{V_n * \cos \alpha_n1}}  $

该极值点处满足
$$
\begin{cases}
\bigtriangledown dH = \lambda \bigtriangledown T(n)  ,\\
T(n) = T\\

\end{cases}
$$
求梯度如下
$$
\begin{cases}
{S_1\sin\alpha_1 \lambda\over{V\cos \alpha_1}^2 } = {{S_1 (V +V_1*\sin \alpha_1)}\over V \cos^2\alpha_1} ,\\
{S_2\sin\alpha_2 \lambda\over{V\cos \alpha_2}^2 } = {{S_2 (V +V_2*\sin \alpha_2)}\over V \cos^2\alpha_2} ,\\
... ,\\
{S_n\sin\alpha_n \lambda\over{V\cos \alpha_n}^2 } = {{S_n (V +V_n*\sin \alpha_n)}\over V \cos^2\alpha_n} ,\\
\end{cases}
$$
约去公因式，可得
$$
\begin{cases}
 \sin \alpha_1 = {V \over \lambda -V_1},\\
 \sin \alpha_2 = {V \over \lambda-V_2},\\
 ...,\\
 \sin\alpha_n = {V \over \lambda-V_n}
 \end{cases}
$$

带入时间不变的等式，$解得\lambda$

进而求解角度

