import os 
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.ticker import ScalarFormatter





def get_data(data_path):
    return pd.read_csv(os.path.join(data_path, 'train_stats.csv'))

def get_args(data_path):
    return json.loads(open(os.path.join(data_path, 'args.json')))

def sci_notation(x, pos):
    """将刻度转为科学计数法（如 5e-1 -> 5×10⁻¹）"""
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exponent
    return r"${0:.0f} \times 10^{{{1}}}$".format(coeff, exponent)

def plot_grouped_results(
    dfs, legends, temps, columns=["train_acc", "train_loss"], 
    title_prefix='CIFAR10', log_scale=False, start_epoch=0,
    end_epoch=300
    ): 
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    axes = axes.flatten()

    for i, temp in enumerate(temps):
        for j, column in enumerate(columns):
            ax_index = i + j * len(temps)
            ax = axes[ax_index]

            x_vals = range(start_epoch, end_epoch)
            y1 = dfs[2 * i][column][start_epoch:end_epoch]
            y2 = dfs[2 * i + 1][column][start_epoch:end_epoch]

            if j == 1:  # train_loss
                ax.semilogy(x_vals, y1, label=legends[2 * i])
                ax.semilogy(x_vals, y2, label=legends[2 * i + 1])
                ax.set_ylabel("Loss (log scale)")

                # show ticks in scientific notation
                ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))

                # show minor ticks
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.5,), numticks=10))

                # using FuncFormatter to format the ticks
                formatter = FuncFormatter(sci_notation)
                ax.yaxis.set_major_formatter(formatter)
                ax.yaxis.set_minor_formatter(formatter)

            else:
                ax.plot(x_vals, y1, label=legends[2 * i])
                ax.plot(x_vals, y2, label=legends[2 * i + 1])
                ax.set_ylabel("Accuracy")

            ax.set_title(f"{title_prefix} T={temp}")
            ax.set_xlabel("Epoch")
            ax.legend()

    plt.tight_layout()
    plt.show()