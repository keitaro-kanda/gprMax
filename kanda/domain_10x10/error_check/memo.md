# Numerical dipression error check
- mesh_001: mesh size is 0.01 m,
- mesh_005: mesh size is 0.05 m, WARNING is occured
    - error message:
    - Potentially significant numerical dispersion. Estimated largest physical phase-velocity error is -2.62% in material 'basalt_6' whose wavelength sampled by 5 cells. Maximum significant frequency estimated as 4.14727e+08Hz

## Theoretical estimation of delay time
model:
- basalt layer is thickness of 5m, $\varepsilon_r = 6$
$$
t = \frac{10 \times \sqrt{6}}{c} \simeq 81.65 \cdots \ (\mathrm{ns})
$$

- transmitting delay: 8.73 ns (at peak)

Observed delay times is estimated to be about 90.38 ns.

## Result of delay time (at peak)
- mesh_001: t=55.6 ns
- mesh_005: t=55.9 ns -> error: 