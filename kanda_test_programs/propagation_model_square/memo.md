# Bottom component
Propagation path:
$$
L_b = 2 \times (h + d \times \sqrt{3} + s \times \sqrt{9})
$$

Propagation time:
$$
\tau_b = \frac{L_b}{c}
$$

# Side component
Propagation path:
$$
L_s = \frac{h}{\cos \theta_1} + \frac{d \sqrt{3}}{\cos \theta_2} + 2\frac{s \sqrt{9}}{\cos \theta_3} + \frac{d\sqrt{3}}{\cos \theta_4} + \frac{h'}{\cos \theta_5}
$$

Snell's law
$$
\begin{split}
&\frac{c}{c/\sqrt{3}} = \frac{\sin \theta_1}{\sin \theta_2} \\
&\sin \theta_2 = \frac{\sin \theta_1}{\sqrt{3}}
\end{split}
$$

$$
\begin{split}
&\frac{c/\sqrt{9}}{c/\sqrt{3}} = \frac{\sin \theta_3}{\sin \theta_4} \\
&\sin \theta_4 = \sqrt{3} \sin \theta_3 = \frac{\sin \theta_1}{\sqrt{3}}
\end{split}
$$

$$
\begin{split}
&\frac{c/\sqrt{9}}{c/\sqrt{3}} = \frac{\sin \theta_3}{\sin \theta_4} \\
&\sin \theta_4 = \sqrt{3} \sin \theta_3 = \frac{\sin \theta_1}{\sqrt{3}}
\end{split}
$$

$$
\begin{split}
&\frac{c/\sqrt{9}}{c/\sqrt{3}} = \frac{\sin \theta_3}{\sin \theta_4} \\
&\sin \theta_4 = \sqrt{3} \sin \theta_3 = \frac{\sin \theta_1}{\sqrt{3}}
\end{split}
$$

$$
\begin{split}
&\frac{c/\sqrt{3}}{c} = \frac{\sin \theta_4}{\sin \theta_5} \\
&\sin \theta_5 = \sqrt{3} \sin \theta_4 = \sin \theta_1
\end{split}
$$

Rewrite propagation path using $\theta_1$:
$$
\begin{split}
L_s &= \frac{h}{\cos \theta_1} + d \sqrt{3} \sqrt{\frac{3}{3 - \sin^2 \theta_1}} + 2 \cdot 3s \frac{3}{\sqrt{9 - \sin^2 \theta_1}} + d \sqrt{3} \sqrt{\frac{3}{3 - \sin^2 \theta_1}} + \frac{h'}{\cos \theta_1} \\
&= \frac{h + h'}{\cos \theta_1} + \frac{6 d}{\sqrt{3 - \sin^2 \theta_1}} + \frac{18 s}{\sqrt{9 - \sin^2 \theta_1}} 
\end{split}
$$