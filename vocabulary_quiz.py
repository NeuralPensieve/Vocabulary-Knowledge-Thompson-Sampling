# %%
import tyro
import math
from dataclasses import dataclass
from models import VocabularyKnowledgeModel
from utils import find_closest, normal_ccdf_area
from pprint import pprint


# %%
@dataclass
class config:
    model_type: str = "bayesian"
    data_file: str = "corpus/english_words.csv"
    mu0: int = 25_000
    std0: int = 12_000
    max_num_words: int = 50_000
    num_iterations: int = 50
    opt_iterations: int = 500
    verbose: bool = False


config = tyro.cli(config)
if config.verbose:
    pprint(config.__dict__)
# %%
# read corpus/english_words.csv which has two columns: word and rank, into two lists: words and ranks
with open(config.data_file, "r") as f:
    words = []
    ranks = []
    for line in f:
        word, rank = line.strip().split(",")
        words.append(word)
        ranks.append(int(rank))


# %%
# Initialize the model
model = VocabularyKnowledgeModel(
    mu0=config.mu0, std0=config.std0, max_num_words=config.max_num_words
)

if config.model_type == "bayesian":
    update = model.update_posterior_bayesian
elif config.model_type == "non-bayesian":
    update = model.update_posterior_non_bayesian
else:
    raise ValueError(f"Unknown model: {config.model_type}")
# %%
x_data = []
y_data = []

for itr in range(config.num_iterations):
    while True:
        sampled_difficulty = model.sample_from_prior()
        sampled_difficulty, index = find_closest(ranks, sampled_difficulty)
        sampled_word = words[index]

        if sampled_difficulty not in model.observed_difficulties:
            break

    # ask the user whether they know the word. if the answer is anything other than 0 or 1, repear the question
    while True:
        if config.verbose:
            prompt = f'Do you know the word \033[91m"{sampled_word}"\033[0m (difficulty: {sampled_difficulty:.0f})? (0 or 1)'
        else:
            prompt = f'Do you know the word \033[91m"{sampled_word}"\033[0m? (0 or 1)'
        response = input(prompt)
        if response in ["0", "1"]:
            response = int(response)
            break
        else:
            print("Please enter either 0 or 1.")

    x_data.append(sampled_difficulty)
    y_data.append(response)

    model.observed_difficulties.append(sampled_difficulty)
    model.observed_responses.append(response)

    # No need to convert to tensor here since update_posterior handles it
    update(x_data, y_data, config.opt_iterations, verbose=config.verbose)

    if config.verbose:
        print("*" * 30)
        print(f"iteration: \033[92m{itr+1}\033[0m")
        print(f"difficulty: {sampled_difficulty:.0f}. Response: {response}.")
        print(f"mu: {model.history_mu[-1]:.0f}. std: {model.history_std[-1]:.0f}.")
        print("*" * 30)

print("\n\n\n")
print(
    f"final mu: {model.history_mu[-1]:.0f}. standard error: {model.history_std[-1] / math.sqrt(config.num_iterations - 1):.0f}."
)
print(f"area under curve: {normal_ccdf_area(model.mu0, model.std0, 0, 50_000):.0f}")

# %%
# ******************************
# iteration: 49
# difficulty: 47623. Response: 0.
# mu: 27339. std: 19335.
# ******************************

# ******************************
# iteration: 49
# difficulty: 38175. Response: 0.
# mu: 31889. std: 14378.
# ******************************
