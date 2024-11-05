import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
from Helmholtz2D_model_tf import Sampler, Helmholtz2D
import os


def plot_gradient_distributions(gradients_res_dict, gradients_bcs_dict, num_layers,plot_folder, xlim=(-3, 3), ylim=(0, 100)):
    fig, axes = plt.subplots(1, num_layers, figsize=(13, 4), sharey=True)
    for idx, ax in enumerate(axes):
        ax.set_title(f'Layer {idx + 1}')
        ax.set_yscale('symlog')

        # Retrieve gradients
        gradients_res = gradients_res_dict[f'layer_{idx + 1}'][-1]
        gradients_bcs = gradients_bcs_dict[f'layer_{idx + 1}'][-1]

        # Plot gradients using kdeplot
        sns.kdeplot(gradients_bcs, ax=ax, bw_adjust=1, fill=False,
                    label=r'$\nabla_\theta \lambda_{u_b} \mathcal{L}_{u_b}$')
        sns.kdeplot(gradients_res, ax=ax, bw_adjust=1, fill=False, label=r'$\nabla_\theta \mathcal{L}_r$')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if idx == 0:  # Show legend only once
            ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_folder + '/gradients.png')

def plot_eigenvalues(eigenvalues_res, eigenvalues_bcs,plot_folder):
    fig_5 = plt.figure(5)
    ax = fig_5.add_subplot(1, 1, 1)
    ax.plot(eigenvalues_res, label='$\mathcal{L}_r$')
    ax.plot(eigenvalues_bcs, label='$\mathcal{L}_{u_b}$')
    ax.set_xlabel('index')
    ax.set_ylabel('eigenvalue')
    ax.set_yscale('symlog')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder + '/eigenvalue.png')


def plot_adaptive_constant(adaptive_constant,plot_folder):
    fig_3 = plt.figure(3)
    ax = fig_3.add_subplot(1, 1, 1)
    ax.plot(adaptive_constant, label='$\lambda_{u_b}$')
    ax.set_xlabel('iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder + '/adaptive_constant.png')


def plot_loss_evolution(loss_res, loss_bcs, plot_folder):
    fig_2 = plt.figure(2)
    ax = fig_2.add_subplot(1, 1, 1)
    ax.plot(loss_res, label='$\mathcal{L}_{r}$')
    ax.plot(loss_bcs, label='$\mathcal{L}_{u_b}$')
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_folder + '/loss.png')


def plot_prediction(x1, x2, U_star, U_pred,plot_folder):
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x1, x2, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Exact $u(x)$')

    plt.subplot(1, 3, 2)
    plt.pcolor(x1, x2, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Predicted $u(x)$')

    plt.subplot(1, 3, 3)
    plt.pcolor(x1, x2, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.savefig(plot_folder + '/solution.png')




def plot_adaptive_constant(adaptive_constant, plot_folder):
    plt.figure()
    plt.plot(adaptive_constant, label='$\lambda_{u_b}$')
    plt.xlabel('Iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'adaptive_constant.png'))


def training_function(a_1=1, a_2=1, lam=1.0, batch_size=128, nIter=100, seed_value=1234):
    tf.random.set_seed(seed_value)  # TensorFlow
    np.random.seed(seed_value)  # NumPy

    parameter_string = 'a_1={:.1f}, a_2={:.1f}, lam={:.1f}'.format(a_1, a_2, lam)
    os.makedirs(parameter_string, exist_ok=True)

    def u(x, a_1, a_2):
        return np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    def u_xx(x, a_1, a_2):
        return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    def u_yy(x, a_1, a_2):
        return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    # Forcing
    def f(x, a_1, a_2, lam):
        return u_xx(x, a_1, a_2) + u_yy(x, a_1, a_2) + lam * u(x, a_1, a_2)

    # Domain boundaries
    bc1_coords = np.array([[-1.0, -1.0],
                           [1.0, -1.0]])
    bc2_coords = np.array([[1.0, -1.0],
                           [1.0, 1.0]])
    bc3_coords = np.array([[1.0, 1.0],
                           [-1.0, 1.0]])
    bc4_coords = np.array([[-1.0, 1.0],
                           [-1.0, -1.0]])

    dom_coords = np.array([[-1.0, -1.0],
                           [1.0, 1.0]])

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC4')
    bcs_sampler = [bc1, bc2, bc3, bc4]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: f(x, a_1, a_2, lam), name='Forcing')

    # Define model
    mode = 'M1'  # Method: 'M1', 'M2', 'M3', 'M4'
    stiff_ratio = False  # Log the eigenvalues of Hessian of losses

    layers = [2, 50, 50, 50, 1]
    model = Helmholtz2D(layers, None, None, bcs_sampler, res_sampler, lam, mode, stiff_ratio)

    # Train model
    model.train(nIter=nIter, batch_size=batch_size)

    # Test data
    nn = 100
    x1 = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x2 = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    x1, x2 = np.meshgrid(x1, x2)
    X_star = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))

    # Exact solution
    u_star = u(X_star, a_1, a_2)
    f_star = f(X_star, a_1, a_2, lam)

    # Predictions
    u_pred = model.predict_u(X_star)
    f_pred = model.predict_r(X_star)

    # Relative error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_f = np.linalg.norm(f_star - f_pred, 2) / np.linalg.norm(f_star, 2)

    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_f: {:.2e}'.format(error_f))

    ### Plot ###

    # Exact solution & Predicted solution
    # Exact solution
    U_star = griddata(X_star, u_star.flatten(), (x1, x2), method='cubic')

    # Predicted solution
    U_pred = griddata(X_star, u_pred.flatten(), (x1, x2), method='cubic')

    plot_prediction(x1, x2, U_star, U_pred, parameter_string)

    loss_res = model.loss_res_log
    loss_bcs = model.loss_bcs_log

    plot_loss_evolution(loss_res, loss_bcs, parameter_string)

    # Adaptive Constant
    adaptive_constant = model.adaptive_constant_log
    plot_adaptive_constant(adaptive_constant, parameter_string)

    # Plot Gradient Distributions
    data_gradients_res = model.dict_gradients_res_layers
    data_gradients_bcs = model.dict_gradients_bcs_layers
    num_hidden_layers = len(layers) - 1
    plot_gradient_distributions(data_gradients_res, data_gradients_bcs, num_hidden_layers, parameter_string)

    # Plot Eigenvalues if applicable
    if stiff_ratio:
        eigenvalues_res = model.eigenvalue_res_log[-1]
        eigenvalues_bcs = model.eigenvalue_bcs_log[-1]
        plot_eigenvalues(eigenvalues_res, eigenvalues_bcs, parameter_string)


if __name__ == '__main__':
    a_1 = 1
    a_2 = 4

    # Parameter
    lam = 1.0

    batch_size=128
    nIter =40001

    seed_value=1

    training_function(
        a_1=a_1,
        a_2=a_2,
        lam=lam,
        batch_size=batch_size,
        nIter=nIter,
        seed_value=seed_value)

    



    
    
    
        
        
    
     
     
    
    
    
    
    

