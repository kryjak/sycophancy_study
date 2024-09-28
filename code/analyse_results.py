import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from axes_and_classes import *
from experiment_list import *
from typing import Tuple, Literal

# Set up the plot style with seaborn
sns.set_style("whitegrid")

# ------------ PLOTTING FUNCTIONS FOR THE EXPERIMENTS ------------
def read_results_from_df(df: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
    # get percentages:
    df_sub = df[df['affirmative_class'] == True]
    affirmative_sycophantic_base = (df_sub['sycophantic_answer'] == df_sub['baseline_answer']).mean()
    affirmative_sycophantic_finetuned = (df_sub['sycophantic_answer'] == df_sub['finetuned_answer']).mean()
    affirmative_truthful_coincidence = (df_sub['sycophantic_answer'] == df_sub['truthful_answer']).mean()

    df_sub = df[df['affirmative_class'] == False]
    non_affirmative_sycophantic_base = (df_sub['sycophantic_answer'] == df_sub['baseline_answer']).mean()
    non_affirmative_sycophantic_finetuned = (df_sub['sycophantic_answer'] == df_sub['finetuned_answer']).mean()
    non_affirmative_truthful_coincidence = (df_sub['sycophantic_answer'] == df_sub['truthful_answer']).mean()

    return affirmative_sycophantic_base, affirmative_sycophantic_finetuned, affirmative_truthful_coincidence, non_affirmative_sycophantic_base, non_affirmative_sycophantic_finetuned, non_affirmative_truthful_coincidence

def read_openended_results_from_df(df: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
    """Open-ended experiments do not have a truthfulness value."""
    # get percentages:
    df_sub = df[df['affirmative_class'] == True]
    affirmative_sycophantic_base = (df_sub['sycophantic_answer'] == df_sub['baseline_answer']).mean()
    affirmative_sycophantic_finetuned = (df_sub['sycophantic_answer'] == df_sub['finetuned_answer']).mean()

    df_sub = df[df['affirmative_class'] == False]
    non_affirmative_sycophantic_base = (df_sub['sycophantic_answer'] == df_sub['baseline_answer']).mean()
    non_affirmative_sycophantic_finetuned = (df_sub['sycophantic_answer'] == df_sub['finetuned_answer']).mean()

    return affirmative_sycophantic_base, affirmative_sycophantic_finetuned, non_affirmative_sycophantic_base, non_affirmative_sycophantic_finetuned 

def plot_single_axis(ax, ax_name, experiment):
    # Unpack data
    is_openeded = True if experiment == 'openended' else False

    df = pd.read_csv(f'data_storage/results_{experiment}_{ax_name}.csv')
    if not is_openeded:
        affirmative_sycophantic_base, affirmative_sycophantic_finetuned, affirmative_truthful_coincidence, non_affirmative_sycophantic_base, non_affirmative_sycophantic_finetuned, non_affirmative_truthful_coincidence = read_results_from_df(df)
    else:
        affirmative_sycophantic_base, affirmative_sycophantic_finetuned, non_affirmative_sycophantic_base, non_affirmative_sycophantic_finetuned = read_openended_results_from_df(df)

    # Data for sycophantic answers (two bars per class)
    sycophantic_data = [
        [affirmative_sycophantic_base, affirmative_sycophantic_finetuned],
        [non_affirmative_sycophantic_base, non_affirmative_sycophantic_finetuned]
    ]
    
    x_labels = ['affirmative', 'non-affirmative']
    x = np.arange(len(x_labels))
    width = 0.35
    
    # Plotting sycophantic answer percentages as bars
    ax.bar(x - width/2, sycophantic_data[0], width, label='base', color='skyblue')
    ax.bar(x + width/2, sycophantic_data[1], width, label='fine-tuned', color='orange')

    # display the percentages on the bars (below the top edge of the bars)
    for i, v in enumerate(sycophantic_data[0]):
        ax.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    for i, v in enumerate(sycophantic_data[1]):
        ax.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Plotting horizontal lines for truthful answer percentages
    if not is_openeded:
        ax.hlines(affirmative_truthful_coincidence, xmin=-0.5, xmax=0.5, color='black', linestyles='dashed', label='truthful coincidence')
        ax.hlines(non_affirmative_truthful_coincidence, xmin=0.5, xmax=1.5, color='black', linestyles='dashed')

    # Setting the title for each subplot
    ax.set_title(f'Axis: {ax_name}', fontsize=16, fontweight='bold')
    
    # Set the x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(labelsize=14)

    # Set the y-axis limits
    ax.set_ylim(0, 1.05)  # Adjust the upper limit as needed

def create_experiment_plot(experiment):
    fig, axes_sub = plt.subplots(1, len(axes), figsize=(18, 6))

    for i, ax_name in enumerate(axes):
        ax = axes_sub[i]
        plot_single_axis(ax, ax_name, experiment)

    # Add a central legend for the whole figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)

    # Set the overall title
    fig.suptitle(f"Sycophancy levels for the '{experiment}' experiment", fontsize=14, fontweight='bold')
    
    # Set the y-axis label
    axes_sub[0].set_ylabel('Proportion of sycophantic answers', fontsize=12, fontweight='bold')

    # # Adjust layout to avoid overlap
    # plt.tight_layout(rect=[0.03, 0.1, 1, 0.93])

    # Save the figure
    plt.savefig(f'data_storage/sycophancy_{experiment}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def create_combined_experiment_plot():
    fig, axes_sub = plt.subplots(3, len(axes), figsize=(18, 18), sharex=True, sharey=True)
    
    for row, experiment in enumerate(experiments):
        for col, ax_name in enumerate(axes):
            ax = axes_sub[row, col]
            plot_single_axis(ax, ax_name, experiment)
            
            # Remove individual titles
            ax.set_title('')
            
            # Add experiment label to the right side of each row
            if col == len(axes) - 1:
                experiment_label = experiment.capitalize().replace('Openended', 'Open-ended')
                ax.text(1.05, 0.5, f'Experiment: {experiment_label}', rotation=-90, 
                        transform=ax.transAxes, va='center', fontsize=18, fontweight='bold')
    
    # Add a central legend for the whole figure
    handles, labels = axes_sub[-1, -1].get_legend_handles_labels()
    # Ensure 'truthful' is included in the legend
    if 'truthful coincidence' not in labels:
        handles.append(plt.Line2D([0], [0], color='black', linestyle='dashed'))
        labels.append('truthful coincidence')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=14)
    
    # Set the overall title
    fig.suptitle("Sycophancy levels for the 'easy', 'hard' and 'open-ended' experiments", fontsize=20, fontweight='bold', y=0.95)
    
    # Set the y-axis label for the leftmost column
    fig.text(0.04, 0.5, 'Proportion of sycophantic answers', va='center', rotation='vertical', fontsize=18, fontweight='bold')
    
    # Add column titles (axis names)
    for col, ax_name in enumerate(axes):
        axes_sub[0, col].set_title(f'Axis: {ax_name}', fontsize=16, fontweight='bold')
    
    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
    # Save the figure
    plt.savefig('data_storage/sycophancy_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# ------------ PLOTTING FUNCTION FOR THE UNDERYLING KNOWLEDGE CHECK ------------
def create_knowledge_check_plot(case: Literal['train', 'test']):
    df = pd.read_csv(f'data_storage/filtering_knowledge_check_{case}.csv')
    # calculate the percentages of truthful answers
    correct_percentages = []

    base = (df['truthful_answer'] == df['baseline_answer']).mean()
    base_again = (df['truthful_answer'] == df['baseline_answer_again']).mean()
    correct_percentages.append(base)
    correct_percentages.append(base_again)

    for axis in axes:
        finetuned_percentage = (df['truthful_answer'] == df[f'{axis}_finetuned_answer']).mean()
        correct_percentages.append(finetuned_percentage)
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set the x-axis ticks and labels
    x = range(len(correct_percentages))
    ax.set_xticks(x)
    ax.set_xticklabels(['baseline', 'baseline again'] + [f'{axis} fine-tuned' for axis in axes], rotation=15, ha='right')
    ax.tick_params(axis="x", labelsize=12)
    
    ax.bar(x, correct_percentages)
    ax.set_title(f'Knowledge check for {case} statements', fontsize=16, fontweight='bold')
    ax.set_ylabel('Proportion of correct answers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    
    # display the percentages on the bars (below the top edge of the bars)
    for i, v in enumerate(correct_percentages):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Set the y-axis limits
    ax.set_ylim(0, 1.05)  # Adjust the upper limit as needed
    
    plt.tight_layout()
    plt.savefig(f'data_storage/knowledge_check_{case}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

#  -------------- PLOTTING FUNCTION FOR THE UNBIASED SYCOPHANCY CHECK -----------------
def create_openended_unbiased_sycophancy_plot():
    df = pd.read_csv('data_storage/openended_unbiased_sycophancy_check.csv')
    
    # Get all columns ending with '_finetuned_answer'
    finetuned_columns = [col for col in df.columns if col.endswith('_finetuned_answer')]
    
    # Calculate the percentages of 'Agree' answers
    sycophantic_data = [(df['baseline_answer'] == '(A)').mean()]
    sycophantic_data.extend((df[col] == '(A)').mean() for col in finetuned_columns)

    # Prepare labels
    x_labels = ['baseline'] + [f'{axis} fine-tuned' for axis in axes]
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(max(10, len(x_labels)), 6))
    
    # Set the x-axis ticks and labels
    x = np.arange(len(x_labels))
    ax.bar(x, sycophantic_data)
    
    # Set the title and labels
    ax.set_title('Preference check for the open-ended unbiased prompts', fontsize=16, fontweight='bold')
    ax.set_ylabel('Rate of agreement with the statement', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15, ha='right')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.tick_params(axis="x", labelsize=12)

    # Display the percentages on the bars
    for i, v in enumerate(sycophantic_data):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Set the y-axis limits
    ax.set_ylim(0, 1.05)  # Adjust the upper limit as needed
    
    plt.tight_layout()
    plt.savefig('data_storage/openended_unbiased_sycophancy_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == '__main__':
    # Create combined plot
    fig = create_combined_experiment_plot()

    for case in ['train', 'test']:
        fig = create_knowledge_check_plot(case)

    fig = create_openended_unbiased_sycophancy_plot()
