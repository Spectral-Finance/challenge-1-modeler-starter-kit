# Challenge 1 Modeler Starter Kit
**Technical Introduction to the 2023 Decentralized Credit Scoring in Web3 Challenge and Spectral Platform** 

![](./assets/challenge-1-hero.png)


## What is Challenge 1?

While traditional finance relies on credit scores to gauge the risk of default, decentralized finance (DeFi) has been largely dependent on over-collateralization (i.e. the equivalent of a secured credit card). An effective on-chain credit score would improve capital efficiency and create a more inclusive, efficient DeFi ecosystem that could one day surpass traditional financial institutions.

Your goal is to predict liquidation (as per binary classification) of an active borrower on Aave v2 Ethereum and Compound v2 Ethereum using PyTorch. Liquidation here includes both:

* Actual liquidation, when a borrower’s health factor drops below the liquidation threshold, triggering a liquidation event; AND
* Technical liquidation, when a borrower’s health factor drops below a specific threshold.


See the [announcement page](https://blog.spectral.finance/challenge-1-credit-scoring-web3/) for more details on the importance of this problem and challenge design.

## Goals
This repo aims to help participants understand the following components of the challenge:
1. Using the Spectral cli
2. The training dataset
3. The submission format
4. ZKML with ezkl


## Status

**This version is for internal testing only and is not suitable for public release.** 



## For Developers


### Project setup and requirements
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

### Spectral cli setup


Confirm the cli is installed by running
```
spectral-cli --help  
```
If you see the help menu, you're good to go!

To configure the CLI you'll first need to sign up for free accounts on **TODO: insert link to Spectral site** and [Alchemy](https://www.alchemy.com/?ref=github.spectral.finance)

```
spectral-cli configure
```
Follow the instructions in terminal to configure the cli.


You should see a message stating "Config has been updated. Spectral CLI is configured!"



### Download Data, Start Modeling, and Prepare Submissions

After configuration, you are ready to begin exploring the dataset and training your model!

[See this notebook for a walkthrough of the entire process](./modeler_starter_kit.ipynb)

## Getting Help
Reach out to us on Discord or Twitter if you have any questions
* [Discord](https://discord.com/invite/Vqwhxva7Y2)
* [Twitter](https://twitter.com/SpectralFi)

If you encounter an issue in this repo [create an issue.](https://github.com/Spectral-Finance/challenge-1-modeler-starter-kit/issues/new)



## Acknowledgements
This work would not be possible without the contributions from the following teams:
* [Alchemy](https://www.alchemy.com/?ref=github.spectral.finance)
* [Cryo](https://github.com/paradigmxyz/cryo?ref=github.spectral.finance)
* [Erigon](https://erigon.ch/?ref=github.spectral.finance)
* [ezkl](https://github.com/zkonduit/ezkl?ref=github.spectral.finance)
* [The Graph](https://thegraph.com/?ref=github.spectral.finance)
* [Messari](https://subgraphs.messari.io/?ref=github.spectral.finance)
* [Reth](https://github.com/paradigmxyz/reth?ref=github.spectral.finance)
* [Transpose](https://www.transpose.io/?ref=github.spectral.finance)