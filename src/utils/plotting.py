import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_loss_over_epoch(values, title, folder):
    plt.figure(figsize=(10, 5))
    plt.plot(values)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{folder}/{title}.png", dpi=300, bbox_inches='tight')
    plt.close()

def get_color_palette(model_list):
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    return {model: custom_colors[i] for i, model in enumerate(model_list)}

def plot_metric_bars(dataframe, split, metric, folder_path):
    model_list = dataframe['model'].unique()
    palette = get_color_palette(model_list)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, strat in zip(axes, dataframe['sampling_strategy'].unique()):

        sns.barplot(
            data=dataframe[dataframe['sampling_strategy'] == strat],
            x='prefix_length',
            y=f'{split}_{metric}',
            hue='model',
            ax=ax,
            edgecolor='black',
            palette=palette,
            errorbar='sd'
        )
        ax.yaxis.grid(True, color='black', alpha=0.4)
        ax.set_title(f'{strat} sampling'.capitalize())
        ax.set_xlabel('Prefix length')
        ax.set_ylabel(metric.replace('_', ' ').capitalize())
        ax.legend_.remove()

    fig.suptitle(f"{split} {metric.replace('_', ' ')}".capitalize())
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title='Model', loc='center', ncol=len(model_list), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'{split}_{metric}.png'), bbox_inches='tight')
    plt.close()