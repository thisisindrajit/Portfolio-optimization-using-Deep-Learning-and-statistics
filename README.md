<img width="1440" alt="Thumbnail" src="https://user-images.githubusercontent.com/43838718/160140118-c355b62a-07bb-4725-b0ca-9f2094900145.png">

# Athena

Portfolio optimization is the process of selecting the best portfolio (asset distribution), out of the set of all portfolios being considered, according to some objective. The objective typically maximizes factors such as expected return, and minimizes costs like financial risk.

Investors want to gain maximum returns from their portfolio while minimizing the risks associated. But purchasing assets without analyzing the fundamentals and merely relying on speculation and market sentiment is a significant problem in portfolio management. This is the problem that the model is trying to solve so that investors can get suggestions from the model and invest wisely.

We intend to propose a novel solution to optimize and enhance portfolio performance using a combination of deep learning and statistical models along with asset fundamentals analysis to obtain maximum returns with minimal risk.


## Uses

- The proposed model can be used to maximize the returns and minimize the risk of a given portfolio (a collection of stocks and other assets) by allocating an optimal weight for each asset in the portfolio.
- It can provide a portfolio that delivers high return per unit risk.
- It can also create a balanced portfolio with many different investments such as stocks, bonds and mutual funds.


## Data collection

For obtaining historical prices of stocks, a python package called YFinance ([yfinance · PyPI](https://pypi.org/project/yfinance/)) is used. 

After the user enters the ticker symbols of all stocks in their portfolio, it is sent to a function which uses APIs from YFinance to get the historical prices of stocks using the ticker symbols provided by the user.

```
def get_data(tickers, period="5y"):

  # checking if the length of the portfolio is greater than the allowed portfolio length
  if port_len > max_port_len:
    print(f'Only {max_port_len} number of assets allowed in portfolio!')
    return None
  
  # download data of ticker symbols
  data = yf.download(tickers, period=period)
  
  # check whether if any ticker symbol has less than minimum required previous days of data
  for i in range(port_len):
    cur_ticker_count = data.iloc[:, i].count()
    print(f'No of rows available for {tickers[i]} is {cur_ticker_count}')

    if cur_ticker_count < min_prev_days:
      print(f'{tickers[i]} has less than {min_prev_days} days of historical price data. Please consider removing the asset or adding some other asset.')
      return None
    
  # having only close prices and removing other data
  data = data.drop("Open", axis = 1)
  data = data.drop("High", axis = 1)
  data = data.drop("Low", axis = 1)
  data = data.drop("Volume", axis = 1)
  data = data.drop("Adj Close", axis = 1)

  return data


df = get_data(tickers, period="5y")
```

Also another API is used to get extra information about the stocks like balance sheets, income statements, financial statements and company information which will later be used for calculating the fundamental analysis scores.

## System Design

![model architecture](https://user-images.githubusercontent.com/43838718/158066436-723127a8-fd99-41aa-bc90-98d6724a3388.png)

### Deep Learning Model

The proposed deep learning model is based on LSTM (Long Short Term Memory) networks. Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem.

The architecture of the proposed deep learning model is shown below:



## Reference Links

### Site links
- [Portfolio Optimization methods](https://blog.quantinsti.com/portfolio-optimization-methods/)

### YouTube links
- [Battle of Portfolio Optimization methods](https://www.youtube.com/watch?v=GW1PASCDOLM&feature=youtu.be)  
- [Portfolio Optimization using Python](https://www.youtube.com/watch?v=xagKMaTjxjk)
- [What is LSTM? Simple explanation of LSTM](https://www.youtube.com/watch?v=LfnrRPFhkuY)
