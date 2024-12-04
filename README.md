# Vocabulary Knowledge Quiz Using Thompson Sampling

This repository contains a Python implementation of a vocabulary knowledge model that uses either a Bayesian or non-Bayesian approach to estimate the difficulty of words based on user responses. The model samples words from a corpus and asks the user whether they know the word, using their responses to update the model's knowledge of word difficulty over multiple iterations.

## Features

- **Thompson Sampling**: Utilizes Thompson sampling to efficiently estimate user's knowledge level.
- **Two Model Types**: Bayesian and non-Bayesian models for word difficulty estimation.
- **Interactive User Input**: Asks the user whether they know a word, with responses affecting the model.
- **Word Difficulty Estimation**: Estimates the difficulty of words using a probabilistic model.
- **Data Input**: Reads word-rank data from a CSV file to sample words for user input.
- **Customizable Configuration**: Control parameters such as number of iterations, model type, and initial priors.

## Requirements
The project requires Python 3.x and several Python packages. You can install them using the `requirements.txt` file.

## Folder Structure

```
.//
├── corpus/
│   ├── data/
│   │   ├── hard.js
│   │   ├── high-frequency-gre.js
│   │   ├── IELTS-4000.txt
│   │   ├── intermediate.js
│   │   ├── popular.txt
│   │   ├── wangyumei-toefl-words.txt
│   │   ├── warm-up.js
│   │   ├── words_alpha.txt
│   │   └── words.db
│   └── english_words.csv
├── data_curation.ipynb
├── LICENSE
├── models.py
├── plots.py
├── README.md
├── requirements.txt
├── simulation.ipynb
├── utils.py
└── vocabulary_quiz.py
```
### Explanation of Main Files

- **`corpus/`**: Contains datasets and word lists used for vocabulary testing.
  - **`data/`**: A collection of various word lists in multiple formats (e.g., JavaScript, text files, and databases).
  - **`english_words.csv`**: A CSV file containing words and their associated ranks. This file is the primary source for word samples in the model.

- **`data_curation.ipynb`**: A Jupyter notebook for data curation, used for preparing and cleaning the word data for use in the model.

- **`simulation.ipynb`**: A Jupyter notebook for running simulations, where the model can be tested interactively with user simulations.

- **`vocabulary_quiz.py`**: The main script for vocabulary knowledge quiz. It prompts users with sampled words and records their responses, which are then used to update the model.