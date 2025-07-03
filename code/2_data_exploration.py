import os
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from dialz import Dataset, SteeringModel, SteeringVector, visualize_activation, get_activation_score

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
df = pd.read_csv("../data/hate_speech_binary.csv").sample(n=500, random_state=42)

def visualise_layers(dataset, scoring_method, fig_name):
    ## Initialize a steering model that activates on layers 10 to 19
    model = SteeringModel(model_name, layer_ids=list(range(10,20)), token=hf_token)

    ## Train the steering vector using the above model and dataset
    vector = SteeringVector.train(model, dataset)

    # Configuration
    n_layers = 30
    n_cols = 5
    n_rows = (n_layers + n_cols - 1) // n_cols  # Ceil division

    # Create subplots
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    # Generate plots
    for layer in range(1, n_layers + 1):
        ax = axes[layer - 1]
        
        # Compute activation scores for this layer
        df['score'] = df['text'].apply(
            lambda x: get_activation_score(
                x, model, vector,
                layer_index=layer,
                scoring_method=scoring_method
            )
        )
        colors = df['label'].map({0: 'blue', 1: 'orange'})
        
        ax.scatter(
            df['text'].str.len(),
            df['score'],
            c=colors, alpha=0.7, s=10
        )
        ax.set_title(f'Layer {layer}')
        ax.set_ylim(-3, 3)

        if layer % n_cols == 1:
            ax.set_ylabel('Score')
        if layer > (n_rows - 1) * n_cols:
            ax.set_xlabel('Text Length')

    # Turn off unused axes
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')

    # Layout adjustment and save
    fig.tight_layout()
    output_path = f'../figs/all_layers_{fig_name}.png'
    fig.savefig(output_path, dpi=300)
    print(f"Saved multi-layer scatter plot to {output_path}")


## Run tests
dataset = Dataset.load_dataset(model_name, 'morality')
visualise_layers(dataset, 'final_token', 'morality_final')
visualise_layers(dataset, 'mean', 'morality_mean')
visualise_layers(dataset, 'max_token', 'morality_max')
visualise_layers(dataset, 'median_token', 'morality_median')

dataset = Dataset.create_dataset(model_name, 
                                 ['hate speech', 'loving words'], 
                                 'You are an example of how someone would respond with ', 
                                 'sentence-starters', 
                                 400)
visualise_layers(dataset, 'final_token', 'lovehate_final')
visualise_layers(dataset, 'mean', 'lovehate_mean')
visualise_layers(dataset, 'max_token', 'lovehate_max')
visualise_layers(dataset, 'median_token', 'lovehate_median')