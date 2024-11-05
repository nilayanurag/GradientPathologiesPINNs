import tensorflow as tf
import keras
import numpy as np
import timeit
import time

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y

class Helmholtz2D(keras.Model):
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, lam, model, stiff_ratio):
        super(Helmholtz2D, self).__init__()
        # Normalization constants
        X, _ = res_sampler.sample(int(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x1, self.sigma_x1 = self.mu_X[0], self.sigma_X[0]
        self.mu_x2, self.sigma_x2 = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # Helmholtz constant
        self.lam = tf.constant(lam, dtype=tf.float32)

        # Mode
        self.model_type = model

        # Record stiff ratio
        self.stiff_ratio = stiff_ratio

        # Adaptive constant
        self.beta = 0.9
        self.adaptive_constant_val = tf.Variable(1.0, trainable=False, dtype=tf.float32)

        # Initialize network layers
        self.layers_dims = layers
        self.network = self.build_network(layers)

        # Logger
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.adaptive_constant_log = []

        # Initialize dictionaries for storing gradients
        self.dict_gradients_res_layers = self.generate_grad_dict(self.layers_dims)
        self.dict_gradients_bcs_layers = self.generate_grad_dict(self.layers_dims)

        # Initialize lists to store eigenvalues if stiff_ratio is True
        if self.stiff_ratio:
            self.eigenvalue_log = []
            self.eigenvalue_bcs_log = []
            self.eigenvalue_res_log = []

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def generate_grad_dict(self, layers):
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict[f'layer_{i + 1}'] = []
        return grad_dict


    def build_network(self, layers):
        model_layers = []
        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            model_layers.append(tf.keras.layers.Dense(layers[l+1], activation=tf.nn.tanh,
                                                      kernel_initializer='glorot_normal'))
        # Output layer
        model_layers.append(tf.keras.layers.Dense(layers[-1], activation=None,
                                                  kernel_initializer='glorot_normal'))
        return tf.keras.Sequential(model_layers)

    def net_u(self, x1, x2):
        x = tf.concat([x1, x2], axis=1)
        u = self.network(x)
        return u

    def net_r(self, x1, x2):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x1, x2])
            x1 = tf.convert_to_tensor(x1)
            x2 = tf.convert_to_tensor(x2)
            u = self.net_u(x1, x2)
            u_x1 = tape.gradient(u, x1) / self.sigma_x1
            u_x2 = tape.gradient(u, x2) / self.sigma_x2
        u_xx1 = tape.gradient(u_x1, x1) / self.sigma_x1
        u_xx2 = tape.gradient(u_x2, x2) / self.sigma_x2
        residual = u_xx1 + u_xx2 + self.lam * u
        del tape
        return residual

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train_step(self, X_bc1_batch, u_bc1_batch,
                   X_bc2_batch, u_bc2_batch,
                   X_bc3_batch, u_bc3_batch,
                   X_bc4_batch, u_bc4_batch,
                   X_res_batch, f_res_batch):
        with tf.GradientTape(persistent=True) as tape:
            # Compute predictions
            u_bc1_pred = self.net_u(X_bc1_batch[:, 0:1], X_bc1_batch[:, 1:2])
            u_bc2_pred = self.net_u(X_bc2_batch[:, 0:1], X_bc2_batch[:, 1:2])
            u_bc3_pred = self.net_u(X_bc3_batch[:, 0:1], X_bc3_batch[:, 1:2])
            u_bc4_pred = self.net_u(X_bc4_batch[:, 0:1], X_bc4_batch[:, 1:2])

            r_pred = self.net_r(X_res_batch[:, 0:1], X_res_batch[:, 1:2])

            # Compute losses
            loss_bc1 = tf.reduce_mean(tf.square(u_bc1_pred - u_bc1_batch))
            loss_bc2 = tf.reduce_mean(tf.square(u_bc2_pred - u_bc2_batch))
            loss_bc3 = tf.reduce_mean(tf.square(u_bc3_pred - u_bc3_batch))
            loss_bc4 = tf.reduce_mean(tf.square(u_bc4_pred - u_bc4_batch))
            loss_bcs = self.adaptive_constant_val * (loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4)

            loss_res = tf.reduce_mean(tf.square(r_pred - f_res_batch))

            loss = loss_res + loss_bcs

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Store layer-wise gradients
        self.save_gradients(tape, loss_bcs, loss_res)

        # Compute eigenvalues if stiff_ratio is True
        if self.stiff_ratio:
            eigenvalues, eigenvalues_bcs, eigenvalues_res = self.compute_eigenvalues(tape, loss, loss_bcs, loss_res)
            self.eigenvalue_log.append(eigenvalues.numpy())
            self.eigenvalue_bcs_log.append(eigenvalues_bcs.numpy())
            self.eigenvalue_res_log.append(eigenvalues_res.numpy())

        del tape  # To free memory

        return loss, loss_bcs, loss_res

    def save_gradients(self, tape, loss_bcs, loss_res):
        # Compute gradients for loss_bcs and loss_res
        gradients_bcs = tape.gradient(loss_bcs, self.trainable_variables)
        gradients_res = tape.gradient(loss_res, self.trainable_variables)

        # Map gradients to layers
        index = 0
        for i, layer in enumerate(self.network.layers):
            num_vars = len(layer.trainable_variables)
            grad_res_layer = gradients_res[index:index + num_vars]
            grad_bcs_layer = gradients_bcs[index:index + num_vars]
            index += num_vars

            # Flatten and concatenate gradients
            grad_res_flat = tf.concat([tf.reshape(g, [-1]) for g in grad_res_layer if g is not None], axis=0)
            grad_bcs_flat = tf.concat([tf.reshape(g, [-1]) for g in grad_bcs_layer if g is not None], axis=0)

            # Store in dictionaries
            self.dict_gradients_res_layers[f'layer_{i + 1}'].append(grad_res_flat.numpy())
            self.dict_gradients_bcs_layers[f'layer_{i + 1}'].append(grad_bcs_flat.numpy())

    def compute_eigenvalues(self, tape, loss, loss_bcs, loss_res):
        # Flatten trainable variables
        params = tf.concat([tf.reshape(var, [-1]) for var in self.trainable_variables], axis=0)

        # Compute Hessian for total loss
        grad = tape.gradient(loss, self.trainable_variables)
        grad_flat = tf.concat([tf.reshape(g, [-1]) for g in grad if g is not None], axis=0)
        hessian = tape.jacobian(grad_flat, self.trainable_variables)
        hessian_matrix = self.flatten_hessian(hessian)

        eigenvalues = tf.linalg.eigvalsh(hessian_matrix)

        # Compute Hessian for loss_bcs
        grad_bcs = tape.gradient(loss_bcs, self.trainable_variables)
        grad_bcs_flat = tf.concat([tf.reshape(g, [-1]) for g in grad_bcs if g is not None], axis=0)
        hessian_bcs = tape.jacobian(grad_bcs_flat, self.trainable_variables)
        hessian_bcs_matrix = self.flatten_hessian(hessian_bcs)
        eigenvalues_bcs = tf.linalg.eigvalsh(hessian_bcs_matrix)

        # Compute Hessian for loss_res
        grad_res = tape.gradient(loss_res, self.trainable_variables)
        grad_res_flat = tf.concat([tf.reshape(g, [-1]) for g in grad_res if g is not None], axis=0)
        hessian_res = tape.jacobian(grad_res_flat, self.trainable_variables)
        hessian_res_matrix = self.flatten_hessian(hessian_res)
        eigenvalues_res = tf.linalg.eigvalsh(hessian_res_matrix)

        return eigenvalues, eigenvalues_bcs, eigenvalues_res

    def flatten_hessian(self, hessian):
        # Flatten the Hessian tensor to a 2D matrix
        hessian_list = []
        for h in hessian:
            h_flat = tf.concat([tf.reshape(h_i, [-1]) for h_i in h if h_i is not None], axis=0)
            hessian_list.append(h_flat)
        hessian_matrix = tf.stack(hessian_list, axis=0)
        return hessian_matrix



    def train(self, nIter=10000, batch_size=128):
        start_time = time.time()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
            X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
            X_bc4_batch, u_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], batch_size)

            # Fetch residual mini-batch
            X_res_batch, f_res_batch = self.fetch_minibatch(self.res_sampler, batch_size)

            # Convert to tensors
            X_bc1_batch = tf.convert_to_tensor(X_bc1_batch, dtype=tf.float32)
            u_bc1_batch = tf.convert_to_tensor(u_bc1_batch, dtype=tf.float32)
            X_bc2_batch = tf.convert_to_tensor(X_bc2_batch, dtype=tf.float32)
            u_bc2_batch = tf.convert_to_tensor(u_bc2_batch, dtype=tf.float32)
            X_bc3_batch = tf.convert_to_tensor(X_bc3_batch, dtype=tf.float32)
            u_bc3_batch = tf.convert_to_tensor(u_bc3_batch, dtype=tf.float32)
            X_bc4_batch = tf.convert_to_tensor(X_bc4_batch, dtype=tf.float32)
            u_bc4_batch = tf.convert_to_tensor(u_bc4_batch, dtype=tf.float32)
            X_res_batch = tf.convert_to_tensor(X_res_batch, dtype=tf.float32)
            f_res_batch = tf.convert_to_tensor(f_res_batch, dtype=tf.float32)

            # Training step
            loss, loss_bcs, loss_res = self.train_step(X_bc1_batch, u_bc1_batch,
                                                       X_bc2_batch, u_bc2_batch,
                                                       X_bc3_batch, u_bc3_batch,
                                                       X_bc4_batch, u_bc4_batch,
                                                       X_res_batch, f_res_batch)

            # Update adaptive constant (if needed)
            if self.model_type in ['M2', 'M4']:
                # Update adaptive constant
                max_grad_res = max([tf.reduce_max(tf.abs(g)) for g in self.dict_gradients_res_layers.values()])
                mean_grad_bcs = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in self.dict_gradients_bcs_layers.values()])
                adaptive_constant_value = max_grad_res / mean_grad_bcs
                self.adaptive_constant_val.assign(adaptive_constant_value * (1.0 - self.beta) + self.beta * self.adaptive_constant_val)

            # Logging
            if it % 10 == 0:
                elapsed = time.time() - start_time
                self.loss_bcs_log.append(loss_bcs.numpy() / self.adaptive_constant_val.numpy())
                self.loss_res_log.append(loss_res.numpy())
                self.adaptive_constant_log.append(self.adaptive_constant_val.numpy())
                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_res: %.3e, Adaptive_Constant: %.2f ,Time: %.2f' %
                      (it, loss.numpy(), loss_bcs.numpy(), loss_res.numpy(), self.adaptive_constant_val.numpy(), elapsed))
                start_time = time.time()


    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
        u_star = self.net_u(X_star[:, 0:1], X_star[:, 1:2])
        return u_star.numpy()

    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
        r_star = self.net_r(X_star[:, 0:1], X_star[:, 1:2])
        return r_star.numpy()




