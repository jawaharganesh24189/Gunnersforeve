import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cells = []

# Title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# ⚽ Arsenal FC Match Prediction & Analysis\n",
        "\n",
        "**Complete Self-Contained ML Pipeline - NO External Dependencies**\n"
    ]
})

notebook['cells'] = cells
with open('arsenal_ml_notebook_standalone.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook created")
