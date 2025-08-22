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
    z0, z1 = pts[:-1], pts[1:] # gives vectorised start/end points for all segments. # segment starts/ends
    v = z1 - z0 # segment vectors
    zA = z0 # start
    zB = z0 + v/3 # 1/3 point
    zC = zB + (v/3) * rot60    # add the equilateral "bump" # peak of bump = rotate middle third by +60°
    zD = z0 + 2*v/3 # 2/3 point
    # Replace each segment A→E with A→B→C→D→E for all segments in parallel
    pts = torch.cat([torch.stack([zA, zB, zC, zD], 1).reshape(-1), z1[-1].unsqueeze(0)]) # rebuild the whole polyline in one go.

# Plot
x, y = pts.real.cpu().numpy(), pts.imag.cpu().numpy()
plt.figure(figsize=(7,7))
plt.plot(x, y, linewidth=0.8)
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.tight_layout()
plt.show()
