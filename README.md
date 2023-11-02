# challenge-1-modeler-starter-kit
**Starter Kit for the 2023 Decentralized Credit Scoring in Web3 Challenge** 

![](./assets/challenge-1-hero.png)


## What is Challenge 1?

While traditional finance relies on credit scores to gauge the risk of default, decentralized finance (DeFi) has been largely dependent on over-collateralization (i.e. the equivalent of a secured credit card). An effective on-chain credit score would improve capital efficiency and create a more inclusive, efficient DeFi ecosystem that could one day surpass traditional financial institutions.

Your goal is to predict liquidation (as per binary classification) of an active borrower on Aave v2 Ethereum and Compound v2 Ethereum. Liquidation here includes both:

* Actual liquidation, when a borrower’s health factor drops below the liquidation threshold, triggering a liquidation event; AND
* Technical liquidation, when a borrower’s health factor drops below a specific threshold.

## Goals
This repo aims to help participants understand the following components of the challenge:
1. Using the Spectral cli
2. The training dataset
3. The submission format
4. ZKML with ezkl


## Status

**This version is for internal testing only and is not suitable for public release.** 


## For Developers


First, clone the repository
```
git clone https://github.com/Spectral-Finance/challenge-1-modeler-starter-kit.git
cd challenge-1-modeler-starter-kit
```

Create and activate a new virtual env
```
python3 -m venv env
source env/bin/activate
```

Install the required packages
```
pip install -r requirements.txt
```

If using an application like PyCharm, be sure to set the python interpreter to the virtual env you just created.
