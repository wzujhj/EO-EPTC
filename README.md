# EO-EPTC

Implementation of "EO-EPTC: End-to-End Original Traffic-Based
Encrypted Proxy Traffic Classification Framework".

### Requirement

- python==3.10.13
- d2l==0.17.5
- numpy==2.3.3
- scikit_learn==1.3.2
- torch==2.0.1

------

### Dataset Format

The dataset comprises multiple web flow records, and each record contains a paired instance of original traffic and proxied traffic. For example

```
dataset
	|---- flow pair
	    |---- original flow
        |---- proxied flow
```

Each TCP packet in flow includes attributes such as direction, packet length, SYN, ACK, and PUSH. For example

```
['c2s','517', '0', '1', '1'], ['s2c','1460', '0', '1', '0']
```

Each TLS packet includes attributes such as direction, record type, and record length.For example

```
['c2s','22:1', '238'], ['s2c','22:2', '122']
```

### How to use

```bash
python main.py
```
