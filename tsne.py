# That's an impressive list of imports.
import numpy
from sklearn.manifold import TSNE

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def tsne(data, train_classes, class_desc, rs):
    result = TSNE(random_state=rs, n_iter=10000).fit_transform(data)
    N = len(result)
    colors = []
    for i in range(N):
        colors.append(int(train_classes[i]))
    colors = numpy.asarray(colors)

    scatter(result, colors, class_desc)
    plt.show()
    # plt.savefig('img/tsne-train.png', dpi=120)


def scatter(x, colors, desc):
    # We choose a color palette with seaborn.
    palette = numpy.array(sns.color_palette("hls", 14))

    # We create a scatter plot.
    f = plt.figure(figsize=(12, 12))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(numpy.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(14):
        # Position of each label.
        xtext, ytext = numpy.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, desc[i], fontsize=10)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
