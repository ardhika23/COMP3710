import torch, math
import matplotlib.pyplot as plt

# Params
order  = 5
scale  = 1.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initial equilateral triangle (closed polyline) as complex points
sr3 = math.sqrt(3)
tri = torch.complex(
    torch.tensor([-0.5,  0.5,  0.0, -0.5], device=device) * scale,
    torch.tensor([-sr3/6, -sr3/6, sr3/3, -sr3/6], device=device) * scale
)

# 60° rotation as a complex constant
rot60 = torch.polar(torch.tensor(1.0, device=device), torch.tensor(math.pi/3, device=device))

# Koch iterations
pts = tri
for _ in range(order):
    z0, z1 = pts[:-1], pts[1:]
    v = z1 - z0
    zA = z0
    zB = z0 + v/3
    zC = zB + (v/3) * rot60    # add the equilateral "bump"
    zD = z0 + 2*v/3
    pts = torch.cat([torch.stack([zA, zB, zC, zD], 1).reshape(-1), z1[-1].unsqueeze(0)])

# Plot
x, y = pts.real.cpu().numpy(), pts.imag.cpu().numpy()
plt.figure(figsize=(7,7))
plt.plot(x, y, linewidth=0.8)
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.tight_layout()
plt.show()
