# README.md

# Ocean Profile Comparison

## Overview

The Ocean Profile Comparison project is designed to facilitate the analysis and visualization of oceanographic data from subocean and CTD (Conductivity, Temperature, Depth) expeditions. This project provides tools for reading, processing, and plotting ocean profile data, enabling researchers to compare and analyze different datasets effectively.

## Project Structure

```
ocean-profile-comparison/
├── data/
│   ├── subocean/
│   │   └── sample_expedition/
│   └── ctd/
│       └── sample_expedition/
├── src/
│   ├── readers/
│   ├── processors/
│   └── plotters/
├── notebooks/
│   └── examples/
├── config/
├── requirements.txt
└── setup.py
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Place your data files in the appropriate directories under `data/subocean/sample_expedition/` and `data/ctd/sample_expedition/`.
2. Use the functions in the `src/readers/` module to read your data.
3. Process the data using the functions in the `src/processors/` module.
4. Visualize the results with the plotting functions in the `src/plotters/` module.

## Examples

Refer to the Jupyter notebooks located in the `notebooks/examples/` directory for practical examples of how to use this project.

## Configuration

Configuration variables for subocean and CTD data processing can be found in the `config/` directory in YAML format.

## License

This project is licensed under the MIT License.