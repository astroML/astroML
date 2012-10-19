import pylab as pl

fig = pl.figure()
ax = fig.add_subplot(111, xticks=[],  yticks=[])

ax.text(0.5, 0.5, "Placeholder", size=40, color='gray', alpha=0.5, rotation=25,
        ha='center', va='center', transform=ax.transAxes)
pl.show()
