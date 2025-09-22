import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

simulations = 10000


tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "PLTR"]
numberofassets = len(tickers)
data = yf.download(tickers, start="2023-01-01", interval="1d")

if isinstance(data.columns, pd.MultiIndex):
    if "Adj Close" in data.columns.levels[0]:
        adjclose = data["Adj Close"]
    else:
        adjclose = data["Close"]
else:
    if "Adj Close" in data.columns:
        adjclose = data["Adj Close"]
    else:
        adjclose = data["Close"]

dailyreturns = adjclose.pct_change().dropna()
meanreturn = dailyreturns.mean()
riskassociated = dailyreturns.std()
annualisedmeanreturn = (1+meanreturn)**252 - 1
annualisedvolatilityassociated = riskassociated * np.sqrt(252)

for ticker in annualisedmeanreturn.index:
    print(f"{ticker} mean return: {annualisedmeanreturn[ticker]:.4f}. standard deviation: {annualisedvolatilityassociated[ticker]:.4f}")

dfreturns = dailyreturns.copy()
covmatrix = dfreturns.cov()
corrmatrix = dfreturns.corr()
annualisedcovmatrix = dailyreturns.cov() * 252 

print("Covariance Matrix:\n", annualisedcovmatrix)
print("\nCorrelation Matrix:\n", corrmatrix)

randomweights = np.random.dirichlet(np.ones(numberofassets), simulations)
portfolioexpectedreturn = np.zeros(simulations)
portfoliovolatility = np.zeros(simulations)
portfolioskewness = np.zeros(simulations)
portfolioexcesskurtosis = np.zeros(simulations)


for i in range(simulations):
    portfolioexpectedreturn[i] = np.dot(randomweights[i], annualisedmeanreturn.values)
    portfoliovolatility[i] = np.sqrt(np.dot(randomweights[i].T, np.dot(annualisedcovmatrix, randomweights[i])))
    
    portfolioreturns = np.dot(dailyreturns.to_numpy(), randomweights[i])
    dailymean = np.mean(portfolioreturns)
    dailystd = np.std(portfolioreturns)

    portfolioskewness[i] = np.mean((portfolioreturns - dailymean)**3) / (dailystd ** 3)
    portfolioexcesskurtosis[i] = (np.mean((portfolioreturns - dailymean)**4) / (dailystd ** 2)**2) - 3



riskfreerate = 0
sharperatio = (portfolioexpectedreturn - riskfreerate) / portfoliovolatility

minvarianceindex = np.argmin(portfoliovolatility)
minvariancereturn = portfolioexpectedreturn[minvarianceindex]
minvariancevolatility = portfoliovolatility[minvarianceindex]

maxsharpeidx = np.argmax(sharperatio)
maxsharpereturn = portfolioexpectedreturn[maxsharpeidx]
maxsharpevol = portfoliovolatility[maxsharpeidx]

 
def weightconstraint(weights): #this is a constraint to ensure sum of weights = 1
    return np.sum(weights) - 1

def returnconstraint(weights, meanreturn, target): #ensures portfolio return = target return
    return np.dot(weights, meanreturn) - target

def objectivefunction(weights, covmatrix, skewnesspenalty, kurtosispenalty, dailyreturns):
    portfolioreturns = np.dot(dailyreturns.to_numpy(), weights)
    mean = np.mean(portfolioreturns)
    std = np.std(portfolioreturns)

    skew = np.mean((portfolioreturns - mean)**3) / (std**3)
    excesskurtosis = np.mean((portfolioreturns - mean)**4) / (std**4) - 3

    annualvol = np.sqrt(weights.T @ covmatrix @ weights)
    skewpen = skewnesspenalty * max(0, -skew) #ensures only negative skew is penalised (we want postive skew)
    kurtosispen = kurtosispenalty * (max(0, excesskurtosis)) #this ensures we punish excess kurtosis (we dont want a postive kutosis value)

    return annualvol + skewpen + kurtosispen

targetreturns = np.linspace(min(portfolioexpectedreturn), max(portfolioexpectedreturn), 100)
frontiervalues = []

for target in targetreturns:
    guess = np.ones(numberofassets) / numberofassets
    constraints = [
        {"type": "eq", "fun": weightconstraint},
        {"type": "eq", "fun": lambda w: returnconstraint(w, annualisedmeanreturn.values, target)}
    ]
    bounds = []
    for i in range(numberofassets):
        bounds.append((0,1)) #ensures for each asset the weight is bounded between 0 and 1 ie no leveraging or short selling

    result = minimize(objectivefunction, guess, args= (annualisedcovmatrix, 0.1, 0.1, dailyreturns), method="SLSQP", bounds=bounds, constraints=constraints)
    
    if result.success:
        vol = np.sqrt(result.x.T @ annualisedcovmatrix @ result.x)
        frontiervalues.append(vol)
    else:
        frontiervalues.append(np.nan)


print(maxsharpereturn*100)
print(maxsharpevol*100)

plt.figure(figsize=(10,6))
plt.scatter(portfoliovolatility, portfolioexpectedreturn, c='blue', alpha=0.5)
plt.plot(frontiervalues, targetreturns, 'r-', linewidth=2, label='Efficient Frontier')
plt.scatter(minvariancevolatility, minvariancereturn, c='red', marker='*', s=200, label='Min Variance')
plt.scatter(maxsharpevol, maxsharpereturn, c='green', marker='*', s=200, label='Max Sharpe')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Expected Return")
plt.title("Random Portfolio Simulation with Efficient Frontier")
plt.legend()
plt.show(),