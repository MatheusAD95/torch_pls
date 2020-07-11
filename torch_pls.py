import torch

class PLSRegression1:
    def __init__(self, n_components, cuda=True):
        self.n_components = n_components
        self.cudaf = lambda x: x.cuda() if cuda else x

    def fit(self, X, Y):
        E = self.cudaf(X - X.mean(dim=0))
        F = self.cudaf(Y - Y.mean(dim=0))
        # X = TP^t + E
        T = self.cudaf(torch.zeros((X.size(0), self.n_components)))
        P = self.cudaf(torch.zeros((X.size(1), self.n_components)))
        W = self.cudaf(torch.zeros((X.size(1), self.n_components)))
        # Y = UQ^t + F
        U = self.cudaf(torch.zeros((Y.size(0), self.n_components)))
        Q = self.cudaf(torch.zeros((Y.size(1), self.n_components)))
        # U = BT
        B = self.cudaf(torch.zeros((self.n_components)))
        normed_matmul = lambda x, y: x@y/(y.T@y)
        for i in range(self.n_components):
            U[:, [i]] = F[:, [0]]
            W[:, [i]] = normed_matmul(E.T, U[:, [i]])
            W[:, [i]] /= W[:, [i]].norm()
            T[:, [i]] = normed_matmul(E,   W[:, [i]])
            Q[:, [i]] = normed_matmul(F.T, T[:, [i]])
            Q[:, [i]] /= Q[:, [i]].norm()
            U[:, [i]] = normed_matmul(F,   Q[:, [i]])
            P[:, [i]] = normed_matmul(E.T, T[:, [i]])
            # orthogonalization of t
            T[:, [i]] *= P[:, [i]].norm()
            W[:, [i]] *= P[:, [i]].norm()
            P[:, [i]] /= P[:, [i]].norm()
            # deflation
            B[[i]] = normed_matmul(U[:, [i]].T, T[:, [i]])
            F -= (B[i]*T[:, [i]])@Q[:, [i]].T
            E -= T[:, [i]]@P[:, [i]].T
        self.P, self.W = P, W
        self.Q, self.B = Q, B
        self.y_mean = self.cudaf(Y.mean(dim=0))
        self.x_mean = self.cudaf(X.mean(dim=0))
        return T.cpu(), U.cpu()

    def transform(self, X):
        E = self.cudaf(X) - self.x_mean
        T = self.cudaf(torch.zeros((X.size(0), self.n_components)))
        for i in range(self.n_components):
            T[:, [i]] = E@self.W[:, [i]]
            E -= T[:, [i]]@self.P[:, [i]].T
        return T.cpu()

    def predict(self, X):
        E = self.cudaf((X - X.mean(dim=0)))
        T = self.cudaf(torch.zeros((X.size(0), self.n_components)))
        y = self.cudaf(torch.zeros((X.size(0), 1)))
        for i in range(self.n_components):
            T[:, [i]] = E@self.W[:, [i]]
            E -= T[:, [i]]@self.P[:, [i]].T
            y[:, [0]] += self.B[i]*T[:, [i]]*self.Q[:, [i]].T
        return T.cpu(), (y + self.y_mean).cpu()
