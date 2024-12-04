import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_estimates(history_mu, history_std, true_mu, true_std):
    """
    Plot the evolution of mu and std estimates.

    """
    iterations = range(len(history_mu))
    plt.figure(figsize=(12, 6))

    # Plot the evolution of mu
    plt.subplot(1, 2, 1)
    plt.plot(iterations, history_mu, label="Estimate of Mu", marker="o")
    plt.axhline(true_mu, color="r", linestyle="--", label="True Mu")
    plt.xlabel("Iteration")
    plt.ylabel("Mu")
    plt.title("Evolution of Mu")
    plt.legend()

    # Plot the evolution of std
    plt.subplot(1, 2, 2)
    plt.plot(iterations, history_std, label="Estimate of Std", marker="o")
    plt.axhline(true_std, color="r", linestyle="--", label="True Std")
    plt.xlabel("Iteration")
    plt.ylabel("Std")
    plt.title("Evolution of Std")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_knowledge_curve(
    observed_difficulties,
    observed_responses,
    estimated_mu,
    estimated_std,
    true_mu,
    true_std,
    max_num_words=25_000,
):
    """
    Plot the estimated probability of knowing words at different difficulty levels.
    """
    difficulties = np.linspace(0, max_num_words, 1000)

    # Calculate estimated probabilities
    estimated_probs = 1 - norm.cdf(difficulties, loc=estimated_mu, scale=estimated_std)

    plt.figure(figsize=(10, 6))
    plt.plot(difficulties, estimated_probs, label="Estimated Knowledge Curve")

    # Calculate true probabilities
    true_probs = 1 - norm.cdf(difficulties, loc=true_mu, scale=true_std)
    plt.plot(difficulties, true_probs, "--", label="True Knowledge Curve")

    # Plot observed points
    for diff, resp in zip(observed_difficulties, observed_responses):
        plt.scatter(diff, resp, color="red", alpha=0.5)

    plt.xlabel("Word Difficulty")
    plt.ylabel("Probability of Knowing")
    plt.title("Knowledge Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def animate_normal_distributions(
    means, stds, fixed_mu, fixed_std, num_points=1000, x_range=(0, 25000)
):
    """
    Creates an animation of normal distributions changing over time using Plotly.

    Args:
        means (list): List of means for the normal distributions.
        stds (list): List of standard deviations for the normal distributions.
        fixed_mu (float): Mean for the true distribution distribution (red).
        fixed_std (float): Standard deviation for the true distribution distribution (red).
        num_points (int): Number of points to use for plotting the distributions.
        x_range (tuple): Range of x values to plot (0 to 25000 by default).

    Returns:
        plotly.graph_objects.Figure: The animated figure.
    """
    if len(means) != len(stds):
        raise ValueError("The lengths of means and stds lists must be equal.")

    # Define the x values and track the maximum y value for the dynamic y-axis.
    x = np.linspace(x_range[0], x_range[1], num_points)
    max_y = 0
    frames = []

    # Compute the true distribution distribution (red)
    fixed_y = (1 / (fixed_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - fixed_mu) / fixed_std) ** 2
    )
    fixed_max_y = fixed_y.max()  # Get the max y-value of the true distribution

    # Generate frames for each time step and calculate max_y
    for i, (mean, std) in enumerate(zip(means, stds)):
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        max_y = max(max_y, y.max())

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=x,
                    y=fixed_y,
                    mode="lines",
                    name="True distribution",
                    line=dict(color="red"),
                ),
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"Estimate at t={i+1}",
                    line=dict(color="blue"),
                ),
            ],
            name=str(i),
        )
        frames.append(frame)

    # Calculate the y-axis range to ensure it accommodates the max of both distributions
    max_plot_y = 1.2 * max(max_y, fixed_max_y)

    # Create the base figure with the initial frame
    initial_y = (1 / (stds[0] * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - means[0]) / stds[0]) ** 2
    )
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=fixed_y,
                mode="lines",
                name="True distribution",
                line=dict(color="red"),
            ),
            go.Scatter(
                x=x,
                y=initial_y,
                mode="lines",
                name="Estimate at t=1",
                line=dict(color="blue"),
            ),
        ],
        layout=go.Layout(
            title="Model estimate of the vocabulary size",
            xaxis=dict(title="x", range=x_range),
            yaxis=dict(title="Density", range=[0, max_plot_y]),  # Dynamic y-axis range
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=500, redraw=True),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
        frames=frames,
    )

    return fig
