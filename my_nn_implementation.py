import numpy as np

class Network:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.weights = [np.random.randn(out_layer_dim, in_layer_dim) for
                        out_layer_dim, in_layer_dim in zip(layers[1:], layers[:-1])]
        self.biases = [np.random.randn(layer_dim, 1) for layer_dim in layers[1:]]
        self.cache = []
        self.dC_dw_all_layers = []
        self.dC_db_all_layers = []

    def forward(self, batch):
        # assumes input data will have examples stacked horizontally (right to left)
        # a for activations
        a, _ = batch
        for i in range(self.num_layers - 1):
            z = np.matmul(self.weights[i], a) + self.biases[i]
            a = self.sigmoid(z)
            self.cache.append((z, a))
        return a

    def SGD(self, instances, labels, num_epochs, lr, batch_size, instances_test=None, labels_test=None):
        instances = instances
        labels = labels
        for epoch in range(num_epochs):
            p = np.random.permutation(instances.shape[1])
            instances = (instances.T[p]).T
            labels = (labels.T[p]).T

            mini_batches = [(instances[:, k: k + batch_size], labels[:, k: k + batch_size]) for k in range(0, instances.shape[1], batch_size)]
            for mini_batch in mini_batches:
                self.cache = []
                self.dC_dw_all_layers = []
                self.dC_db_all_layers = []
                dummy = np.zeros([1])
                self.cache.append((dummy, mini_batch[0]))
                predictions = self.forward(mini_batch)
                self.backward(predictions, mini_batch)
                self.step(lr)
                num_correct_predictions = self.evaluate(instances_test, labels_test)
                #print(num_correct_predictions)
            if (instances_test is not None):
                print("Epoch {0}: {1} / {2}".format(epoch,num_correct_predictions, instances_test.shape[1]))

    def backward(self, predictions, batch):
        _, labels = batch
        final_layer_dC_dz = (predictions - labels) * self.sigmoid_prime(self.cache[-1][0])
        dC_dz_curr_layer = final_layer_dC_dz

        for idx, (z, a) in reversed(list(enumerate(self.cache[:-1]))):
            dC_db_curr_layer = dC_dz_curr_layer
            dC_dw_curr_layer = []
            for a_col, dC_dz_col in zip(a.T, dC_dz_curr_layer.T):
                a_col = np.expand_dims(a_col, axis = 1)
                dC_dz_col = np.expand_dims(dC_dz_col, axis = 1)
                dC_dw_curr_layer.append(np.matmul(dC_dz_col, a_col.T))

            self.dC_dw_all_layers.insert(0, dC_dw_curr_layer)
            self.dC_db_all_layers.insert(0, dC_db_curr_layer)

            if idx == 0:
                break

            weight_mat = self.weights[idx]
            dC_dz_curr_layer = np.matmul(weight_mat.T, dC_dz_curr_layer) * self.sigmoid_prime(z)

    def step(self, lr):
        for idx, (weight_nablas, biases_nablas) in enumerate(zip(self.dC_dw_all_layers, self.dC_db_all_layers)):
            dC_dw_all_layers_avg = sum(weight_nablas) / len(weight_nablas)
            dC_db_all_layers_avg = np.expand_dims(np.sum(biases_nablas, axis=1) / biases_nablas.shape[1], axis=1)
            self.weights[idx] = self.weights[idx] - lr * dC_dw_all_layers_avg
            self.biases[idx] = self.biases[idx] - lr * dC_db_all_layers_avg

    def evaluate(self, instances, labels):
        dummy = []
        batch = (instances, dummy)
        results = self.forward(batch)
        results = list(np.argmax(results, axis=0))
        return sum([x == y for x,y in zip(results, labels)])

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
