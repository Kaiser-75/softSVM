import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from cvxopt import matrix, solvers

class SoftSVM:
    def __init__(self, file_path, C=1.0):
        self.df = pd.read_excel(file_path)
        self.X = self.df.iloc[:, :-1].values
        self.y = self.df.iloc[:, -1].values.astype(float).reshape(-1, 1)
        self.C = C
        self.lmbda = None
        self.w = None
        self.w_offset = None
        self.sv_count = 0
        self.mis_count = 0
        self.margin = 0
        self.solve()
        self.count_sv_and_mis()
        self.calculate_margin()

    def solve(self):
        m, _ = self.X.shape
        Xy = self.y * self.X
        H = np.dot(Xy, Xy.T)
        P = matrix(H)
        q = matrix(-np.ones((m, 1)))
        G = np.vstack((-np.eye(m), np.eye(m)))
        h = np.hstack((np.zeros(m), np.ones(m) * self.C))
        G, h = matrix(G), matrix(h)
        A = matrix(self.y.reshape(1, -1))
        b_ = matrix(np.zeros(1))
        sol = solvers.qp(P, q, G, h, A, b_)
        self.lmbda = np.array(sol['x']).flatten()
        w_ = np.sum((self.lmbda[:, None] * self.y) * self.X, axis=0)
        eps = 1e-5
        sv = np.where((self.lmbda > eps) & (self.lmbda < self.C - eps))[0]
        if len(sv) > 0:
            w_offset_vals = [self.y[i_] - np.dot(w_, self.X[i_]) for i_ in sv]
            w_offset_val = np.mean(w_offset_vals)
        else:
            i_ = np.argmax(self.lmbda)
            w_offset_val = self.y[i_] - np.dot(w_, self.X[i_])
        self.w = w_
        self.w_offset = w_offset_val

    def count_sv_and_mis(self):
        eps = 1e-5
        sv = np.where(self.lmbda > eps)[0]
        self.sv_count = len(sv)
        preds = np.sign(np.dot(self.X, self.w) + self.w_offset)
        mis = np.where(preds.reshape(-1, 1) != self.y)[0]
        self.mis_count = len(mis)

    def calculate_margin(self):
        self.margin = 2 / np.linalg.norm(self.w)

def plot_solution(model, ax, title):
    X = model.X
    y = model.y.flatten()
    pos_mask = (y == 1)
    neg_mask = (y == -1)
    ax.scatter(X[pos_mask, 0], X[pos_mask, 1], c='red', edgecolors='k', label='class1')
    ax.scatter(X[neg_mask, 0], X[neg_mask, 1], c='blue', edgecolors='k', label='class2')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.w[0] * xx + model.w[1] * yy + model.w_offset
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    ax.contour(xx, yy, Z, levels=[1], colors='teal', linestyles='--', linewidths=2)
    ax.contour(xx, yy, Z, levels=[-1], colors='teal', linestyles='--', linewidths=2)
    eps = 1e-5
    sv_idx = np.where(model.lmbda > eps)[0]
    ax.scatter(X[sv_idx, 0], X[sv_idx, 1], s=120, color='black', marker='x', label="Support Vectors")
    preds = np.sign(np.dot(X, model.w) + model.w_offset)
    mis = np.where(preds != y)[0]
    if len(mis) > 0:
        ax.scatter(X[mis, 0], X[mis, 1], s=80, facecolors='none', edgecolors='lime', marker='o', label="Misclassified")
    decision_boundary_handle = mlines.Line2D([], [], color='black', linewidth=2, label="Decision Boundary")
    margin_boundary_handle = mlines.Line2D([], [], color='teal', linestyle='--', linewidth=2, label="Margin Boundary")
    ax.set_title(f"{title}\nPts in margin or misclass={model.mis_count}, Margin={model.margin:.3f}")
    ax.legend(handles=[decision_boundary_handle, margin_boundary_handle,
                       mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label="class1"),
                       mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label="class2"),
                       mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=8, label="Support Vectors"),
                       mlines.Line2D([], [], color='lime', marker='o', markerfacecolor='none', linestyle='None', markersize=8, label="Misclassified")],
              loc='lower left')

def plot_efficiency():
    data = {
        'Sample_Size': [50, 100, 200, 500, 1000, 10000],
        'CVXOPT_Time': [0.014000892639160156, 0.02499675750732422, 0.03999924659729004, 
                        0.10799980163574219, 0.611, 385.8],
        'SMO_Time': [0.0010006427764892578, 0.0010025185614356, 0.0010020732879638672, 
                     0.0019996166229248047, 0.004, 0.045]
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df['Sample_Size'], df['CVXOPT_Time'], label='CVXOPT', marker='o', color='blue')
    plt.plot(df['Sample_Size'], df['SMO_Time'], label='SMO', marker='x', color='red')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Computational Efficiency: CVXOPT vs SMO')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, c_ in enumerate([0.1, 100]):
        model = SoftSVM("Proj2DataSet.xlsx", c_)
        plot_solution(model, axes[i], f"C={c_}")
    plt.tight_layout()
    plt.show()
    plot_efficiency()