# ST-machine-learning

## 1. Logic Gate using Logistic Regression

```python
x_data = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
t_data = np.array([1, 0, 0, 0])

logic_gate = LogicGate('AND_Gate', x_data, t_data)

logic_gate.train()
logic_gate.predict(np.array([1, 0]))
```

```shell
# result

Initial error value =  3.167790997444477
step =  0 error value =  3.1384252003915627
step =  400 error value =  1.4700016111676857
step =  800 error value =  1.1071127156150031
step =  1200 error value =  0.8962369925505341
step =  1600 error value =  0.7551737757161494
step =  2000 error value =  0.6529534217380061
step =  2400 error value =  0.5750252523551351
step =  2800 error value =  0.5134871445786156
step =  3200 error value =  0.46360580995934586
step =  3600 error value =  0.4223423951346208
step =  4000 error value =  0.387643183072401
step =  4400 error value =  0.3580643821537696
step =  4800 error value =  0.3325586682771706
step =  5200 error value =  0.31034661878397024
step =  5600 error value =  0.2908356524756517
step =  6000 error value =  0.27356697282496717
step =  6400 error value =  0.25817974009983513
step =  6800 error value =  0.24438622998762977
step =  7200 error value =  0.23195421419204176
step =  7600 error value =  0.22069421401091036
step =  8000 error value =  0.21045011746763412
(array([0.05954009]), 0)
```