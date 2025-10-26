let’s go **deep into your Day 1: Python for Machine Learning** roadmap.

The goal today is to **get your Python environment ready, understand the key libraries (NumPy, Pandas, Matplotlib, Seaborn)**, and complete a few simple **hands-on ML-style tasks** so you can confidently handle datasets tomorrow.

---

## ⚙️ **Step 0: Setup Your Environment**

### 🔧 Install Required Tools

Open a terminal (or VS Code terminal) and run:

```bash
# Install Python packages
pip install numpy pandas matplotlib seaborn jupyter
```

Then start a Jupyter notebook:

```bash
jupyter notebook
```

👉 You’ll write and run all your code there.

---

## 🐍 **Step 1: Core Python Refresher**

Learn the basics (if you already know, skim this in 30–45 min).

### 🔹 Variables & Data Types

```python
x = 10          # integer
name = "Veera"  # string
pi = 3.14       # float
is_active = True  # boolean
```

### 🔹 Lists, Dictionaries, Loops, Functions

```python
# List
nums = [1, 2, 3, 4]
for n in nums:
    print(n * 2)

# Dictionary
student = {"name": "Veera", "score": 95}
print(student["name"])

# Function
def square(num):
    return num ** 2

print(square(5))
```

**Practice**

* Write a function to calculate factorial of a number.
* Create a dictionary with student names and marks, print average marks.

📘 *Resources:*

* [Python Crash Course – FreeCodeCamp (4 hrs)](https://www.youtube.com/watch?v=LHBE6Q9XlzI)
* [Kaggle Python Course](https://www.kaggle.com/learn/python)

---

## 🔢 **Step 2: NumPy – Numerical Operations**

NumPy is the **foundation for ML math** — efficient arrays, vectorized operations, statistics.

### ⚡ Basics

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr.mean())   # mean
print(arr.std())    # standard deviation
print(np.corrcoef(arr, arr))  # correlation
```

### ⚡ Matrix Operations

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))  # matrix multiplication
print(np.linalg.inv(A))  # inverse
```

**Practice**

* Create a NumPy array of random numbers (0–10).
* Compute mean, median, variance.
* Perform element-wise addition and multiplication.

📘 *Resource:* [NumPy Tutorial – W3Schools](https://www.w3schools.com/python/numpy_intro.asp)

---

## 🧾 **Step 3: Pandas – Data Manipulation**

Pandas is used to load, clean, and analyze datasets — your bread and butter in ML.

### ⚡ Load Dataset

```python
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
df.head()
```

### ⚡ Inspect Data

```python
df.info()
df.describe()
df['species'].value_counts()
```

### ⚡ Data Operations

```python
# Selecting columns
df['sepal_length'].mean()

# Filtering rows
df[df['species'] == 'setosa']

# Adding new column
df['sepal_ratio'] = df['sepal_length'] / df['sepal_width']
```

**Practice**

* Load the Iris dataset.
* Find mean and std dev of each column.
* Filter rows where `sepal_length > 5.0`.
* Save cleaned data using `df.to_csv("iris_clean.csv", index=False)`.

📘 *Resource:* [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas)

---

## 📊 **Step 4: Matplotlib & Seaborn – Visualization**

Visualization helps understand data patterns quickly.

### ⚡ Matplotlib Example

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]

plt.plot(x, y)
plt.title("Simple Line Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

### ⚡ Seaborn Example (easier, prettier)

```python
import seaborn as sns

sns.scatterplot(x="sepal_length", y="sepal_width", data=df, hue="species")
plt.title("Iris Sepal Dimensions")
plt.show()
```

### ⚡ More Plot Types

```python
sns.histplot(df['petal_length'])
sns.boxplot(x='species', y='sepal_length', data=df)
```

**Practice**

* Plot a histogram of `sepal_length`.
* Create a boxplot of `petal_width` by species.
* Create a correlation heatmap:

  ```python
  sns.heatmap(df.corr(), annot=True)
  ```

📘 *Resource:* [Seaborn Gallery](https://seaborn.pydata.org/examples/)

---

## 🧠 **Step 5: Day-1 Mini Project**

Combine everything you learned:

### 🪄 Project: “Iris Dataset Quick Analysis”

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Basic info
print(df.describe())

# Compute mean and correlation
print("Mean Sepal Length:", df['sepal_length'].mean())
print("Correlation Matrix:\n", df.corr())

# Visualize
sns.pairplot(df, hue="species")
plt.show()
```

✅ **Outcome:**
You’ve cleaned, analyzed, and visualized data — this is the core skill ML engineers use daily before model training.

---

## 🧩 **Day 1 Summary Checklist**

| Skill         | Task                        | Done? |
| ------------- | --------------------------- | ----- |
| Python Basics | Variables, Loops, Functions | ☐     |
| NumPy         | Mean, Std, Correlation      | ☐     |
| Pandas        | Load & Analyze Dataset      | ☐     |
| Visualization | Matplotlib + Seaborn        | ☐     |
| Mini Project  | Iris Analysis               | ☐     |

---
