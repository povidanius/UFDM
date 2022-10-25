from tbparse import SummaryReader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


def get_mean(df, tag):
 final_iteration = df['step'] == 2520
 yreg1 = df['tag'] == tag
 idx = np.logical_and(final_iteration.to_numpy() , yreg1.to_numpy() )
 aaa = df[idx]['value'].to_numpy()
 return np.mean(aaa), np.var(aaa)

log_dir = "./runs"
reader = SummaryReader(log_dir)
df = reader.scalars

y1,sy1 = get_mean(df,'yReg1/train')
y2,sy2 = get_mean(df,'yReg2/train')
y3,sy3 = get_mean(df,'yReg3/train')

x1,sx1 = get_mean(df,'xReg1/train')
x2,sx2 = get_mean(df,'xReg2/train')
x3,sx3 = get_mean(df,'xReg3/train')

fig, ax = plt.subplots()

trans1 = Affine2D().translate(0.0, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.01, 0.0) + ax.transData
"""
ax.errorbar([1,2,3], [y1,y2,y3], yerr=[sy1,sy2, sy3],color='b',fmt='-',transform=trans1, alpha=0.4)
ax.errorbar([1,2,3], [x1,x2,x3], yerr=[sx1,sx2, sx3],color='o',fmt='-',transform=trans2, alpha=0.4)
plt.savefig('/home/tank/Downloads/art/Journal/elsarticle/experiments/aib/aib.png')
plt.show()
"""

print([y1,y2,y3])
print([x1,x2,x3])

ax.errorbar([x1, x2, x3], [y1,y2,y3], yerr=[sy1,sy2, sy3],color='b',fmt='-',transform=trans1, alpha=0.4)
plt.show()


breakpoint()