
# UKBFound: A Foundation Model for Multi-Disease Prediction and Individual Risk Assessment Based on UK Biobank Data

## Overview
UKBFound is a comprehensive foundation model designed to predict and assess health risks across 1,560 diseases using rich multimodal data from the UK Biobank. This model leverages basic information, lifestyle, measurements, environmental factors, genetics, and imaging data to provide superior predictive accuracy and uncover potential connections among multiple risk factors and diseases.

## Features
- **Multi-Disease Prediction**: Predict risks for 1,560 diseases simultaneously.
- **Individual Risk Assessment**: Assess health risks based on a comprehensive set of variables.
- **Multimodal Data Integration**: Utilize data from various sources including lifestyle, measurements, genetics, and imaging.
- **High Predictive Performance**: Achieve significant improvements in predictive accuracy for a vast majority of disease types.
- **Connections among risk factors and diseases**: Offer a broader perspective on health and multimorbidity mechanisms.
- **Interactive Platform**: Explore detailed results and variable importance through our [interactive platform](https://stardustctrl.shinyapps.io/ukbfound/).

## Repository Structure
This repository is organized into five main parts:

1. **Data Preprocessing**: Cleaning and preparing the data.
2. **Missing Data Processing**: Handling missing data to ensure robust model performance.
3. **Disease Diagnosis**: Predicting various diseases.
4. **Risk Assessment**: Assessing individual health risks.
5. **Others**: The impact of various risk factors and the interrelationships between multiple diseases, comparative analysis and figures.

--------
## Getting Started

### Prerequisites
- Python 3.7 or higher
- R 4.3 or higher
- Required Python libraries (specified in `requirements.txt`)
- Required R packages:
    - tidyverse 2.0.0 
    - data.table 1.14.8
    - missRanger 2.4.0
    - scales 1.2.1
    - forcats 1.0.0
    - ggsci 3.0.0
    - ggpubr 0.6.0
    - optparse 1.7.3
    - plyr 1.8.9
    - zip 2.3.0
    - nFactors 2.4.1.1

### Installation
Clone the repository to your local machine:
```sh
git clone https://github.com/kannyjyk/UKBFound.git
cd UKBFound
```

Install the required Python libraries:
```sh
pip install -r requirements.txt
```

### Usage

#### Data Preprocessing
Navigate to the `scripts/data_preprocessing` folder and run the preprocessing scripts.

#### Missing Data Processing
Handle missing data using the scripts in the `scripts/missing_data_processing` folder.

#### Disease Diagnosis
Run the disease prediction models from the `scripts/disease_diagnosis` folder.

#### Risk Assessment
Assess individual health risks using the models in the `scripts/risk_assessment` folder.

#### Risk Factors and Multimorbidity Analysis
Analyze risk factors and the relationships between multiple diseases using the scripts in the `scripts/disease_diagnosis` and `scripts/plot` folders.

#### Comparative Analysis
Conduct comparative analysis using the scripts in the `scripts/comparative_analysis` folder.

#### Figures
Generate figures in the manuscript using the scripts in the `scripts/plot` folder.

### Machine Learning Methods and Intermediate Results
The parameters, weights, and intermediate results used for plotting are available on Zenodo. You can access them via the following link: [Model parameters and outcomes of "UKBFound: A Foundation Model for Multi-Disease Prediction and Individual Risk Assessment Based on UK Biobank Data"](https://sandbox.zenodo.org/records/73649).

------
## Contributing
We welcome contributions to improve UKBFound. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## Acknowledgements
We thank the UK Biobank for providing the data and all contributors.

