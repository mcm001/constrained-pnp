from constrained_pnp import Strategy, solve_planar_pnp

# Camera extrinsics matrix
K = 1

# World points
world_points = 1

# Image points
image_points = 1

print(solve_planar_pnp(Strategy.NAIVE, 2, 3, 4))
print(solve_planar_pnp(Strategy.POLYNOMIAL, 2, 3, 4))