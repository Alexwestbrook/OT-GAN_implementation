import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Flatten, Reshape, UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import os
import time
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.axes as ax


# GLU activation adapted from "https://github.com/porcelainruler/Gated-Linear
# -Unit-Activation-Implementation-TF/blob/master/GLU.py"
class GLU(Model):
    def __init__(self, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim
        self.sig = tf.sigmoid

    # Function to Slice Tensor Equally along Last Dim
    def equal_slice(self, x):
        ndim = len(x.shape)
        slice_idx = x.shape[self.dim] // 2
        if ndim == 2:
            linear_output = x[:, :slice_idx]
            gated_output = x[:, slice_idx:]
        elif ndim == 3:
            linear_output = x[:, :, :slice_idx]
            gated_output = x[:, :, slice_idx:]
        elif ndim == 4:
            linear_output = x[:, :, :, :slice_idx]
            gated_output = x[:, :, :, slice_idx:]
        elif ndim == 5:
            linear_output = x[:, :, :, :, :slice_idx]
            gated_output = x[:, :, :, :, slice_idx:]
        else:
            raise ValueError(
                "This GLU Activation only support for Dense, 1D, 2D, 3D Conv, "
                "but the Input's Dim is={}".format(ndim))
        # Return the 2 slices
        return linear_output, gated_output

    def call(self, inputs, **kwargs):
        assert inputs.shape[self.dim] % 2 == 0
        # Slicing the Tensor in 2 Halfs
        lin_out_slice, gated_out_slice = self.equal_slice(inputs)
        # Applying Sigmoid Activation to 2nd Slice
        siggat_out_slice = self.sig(gated_out_slice)
        # Returning Element-wise Multiply of two Slices
        return lin_out_slice * siggat_out_slice


class Generator(Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(2*7*7*128)
        self.activation = GLU()
        self.reshape = Reshape((7, 7, 128))
        self.upsample = UpSampling2D((2, 2))
        self.conv1 = Conv2D(filters=2*64, kernel_size=(3, 3), strides=1,
                            padding='same')
        self.conv2 = Conv2D(filters=2*32, kernel_size=(3, 3), strides=1,
                            padding='same')
        self.conv3 = Conv2D(filters=1, kernel_size=(3, 3), strides=1,
                            padding='same', activation='tanh')

    def call(self, inputs, training=False):
        x = self.activation(self.dense(inputs))
        x = self.reshape(x)
        x = self.upsample(x)
        x = self.activation(self.conv1(x))
        x = self.upsample(x)
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x


class Critic(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                            padding='same', activation=tf.nn.crelu)
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=2,
                            padding='same', activation=tf.nn.crelu)
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=2,
                            padding='same', activation=tf.nn.crelu)
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = x / tf.norm(x, axis=1, keepdims=True)
        return x


def training_otgan(dataset, epochs, img_shape=(28, 28, 1), batch_size=128,
                   n_gen_per_critic=3, latent_dim=128, g_learn_rate=0.001,
                   c_learn_rate=0.001, sink_iter=10, entropy_reg=1.0,
                   directory='./', use_critic=True, verbose=True, patience=5):
    """
    Training loop for OT-GAN.

    Arguments
    ---------
    dataset (tf.data.Dataset): entire image dataset, not batched
    img_shape (tuple of ints): shape of images, must have a dimension for
        channels
    epochs (int): number of epochs for training
    batch_size (int): number of images per batch for each training step, must
        be even
    n_gen_per_critic (int): number of generator training steps per critic
        training step
    latent_dim (int): dimension of the latent space from which to sample
        generator inputs
    g_learn_rate (float): learning rate of the generator
    c_learn_rate (float): learning rate of the critic
    sink_iter (int): number of iterations for sinkhorn algorithm
    entropy_reg (float): regularization term in sinkhorn algorithm
    directory (str): path to directory in which to store training logs
    use_critic (bool): indicates whether to use a critic representation of
        images for optimal transport
    patience (int): number of epochs to wait for improvement before applying
        earlystooping
    verbose (bool): whether to print messages during training progress
    """
    # Check that batch can be divided in two equal sized batches
    assert batch_size % 2 == 0
    # Build directory for experiment
    if not os.path.isdir(directory):
        os.mkdir(directory)
    if not os.path.isdir(os.path.join(directory, 'Generated_images')):
        os.mkdir(os.path.join(directory, 'Generated_images'))
    if not os.path.isdir(os.path.join(directory, 'Best_weights_generator')):
        os.mkdir(os.path.join(directory, 'Best_weights_generator'))
    # Shuffle and batch dataset. It will automatically be reshuffled and
    # batched at each call. If batch_size doesn't divide the number of
    # samples, the remaining samples will not be seen during the epoch.
    dataset = dataset.shuffle(
        buffer_size=dataset.cardinality(),
        reshuffle_each_iteration=True
    ).batch(batch_size, drop_remainder=True)

    # Initialize generator and critic along with their optimizer
    generator = Generator()
    generator.build(input_shape=(None, latent_dim))
    g_optimizer = Adam(learning_rate=g_learn_rate)
    if use_critic:
        critic = Critic()
        critic.build(input_shape=(None,) + img_shape)
        c_optimizer = Adam(learning_rate=c_learn_rate)
    else:
        critic = None
        c_optimizer = None

    # Training initializers
    global_step = 0
    epochs_without_improvement = 0
    min_g_loss = np.inf
    # Training loop
    for epoch in range(epochs):
        # Earlystopping
        if epochs_without_improvement >= patience:
            if verbose:
                print(f'Earlystopping after {patience} epochs without '
                      'generator loss improvement')
            break
        if verbose:
            print(f'Epoch: {epoch}')
        # Initialize epoch logs
        start_time = time.time()
        c_step, g_step = 0, 0
        c_tot_loss, g_tot_loss = 0, 0
        # Consider all batches in one epoch
        for data_batch in dataset:
            # Train generator n_gen_per_critic times before training critic
            if global_step % (n_gen_per_critic + 1) == 0:
                c_loss = train_critic(
                    generator, critic, c_optimizer, data_batch,
                    batch_size, latent_dim, sink_iter, entropy_reg
                )
                c_tot_loss += c_loss
                c_step += 1
            else:
                g_loss, gen_images = train_generator(
                    generator, critic, g_optimizer, data_batch,
                    batch_size, latent_dim, sink_iter, entropy_reg
                    )
                g_tot_loss += g_loss
                g_step += 1
            global_step += 1
        # Compute average loss for critic and generator over the epoch
        c_avg_loss = c_tot_loss / c_step
        g_avg_loss = g_tot_loss / g_step
        # Compute epoch time
        epoch_time = time.time() - start_time
        # Record logs in file, and optionally print them
        log_message = (f'critic loss: {c_avg_loss} - generator_loss: '
                       f'{g_avg_loss} - time: {epoch_time}\n')
        if verbose:
            print(log_message)
        with open(os.path.join(directory, 'training_logs.txt'), 'a') as f:
            f.write(log_message)
        # Save last generated images of epoch
        np.save(os.path.join(directory, 'Generated_images',
                             f'gen_images_ep{epoch}'),
                gen_images)
        # Save best model checkpoint
        if min_g_loss > g_avg_loss:
            epochs_without_improvement = 0
            min_g_loss = g_avg_loss
            generator.save_weights(
                os.path.join(directory, 'Best_weights_generator', 'checkpoint')
            )
        else:
            epochs_without_improvement += 1
    # Save models
    generator.save(os.path.join(directory, 'Generator'))
    if use_critic:
        critic.save(os.path.join(directory, 'Critic'))
    # Record hyperparameters and options for training
    with open(os.path.join(directory, 'Experiment_info.txt'), 'w') as f:
        f.write(f'epochs: {epochs}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'n_gen_per_critic: {n_gen_per_critic}\n')
        f.write(f'latent_dim: {latent_dim}\n')
        f.write(f'g_learning_rate: {g_learn_rate}\n')
        f.write(f'c_learning_rate: {c_learn_rate}\n')
        f.write(f'sink_iter: {sink_iter}\n')
        f.write(f'entropy_reg: {entropy_reg}\n')
        f.write(f'use_critic: {use_critic}\n')


def train_critic(generator, critic, c_optimizer, data_batch,
                 batch_size, latent_dim, sink_iter, entropy_reg):
    """
    Training step of the critic.

    Arguments
    ---------
    generator (Generator): model generating images from latent vector
    critic (Critic): model representing images in abstract feature space
    c_optimizer (tf.keras.optimizer): optimizer for the critic
    data_batch (tf.Tensor): batch of batch_size images from data distribution
    batch_size (int): number of image samples in batch, must be even
    latent_dim (int): dimension of the latent space from which to sample
        generator inputs
    sink_iter (int): number of iterations for sinkhorn algorithm
    entropy_reg (float): regularization term in sinkhorn algorithm
    """
    if critic is None:
        return 0
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # Generate 2 batches of fake images
    gen_batch = generator(random_latent_vectors)
    # Separate batches in 2 equal parts
    gen_batch1, gen_batch2 = tf.split(gen_batch, 2)
    data_batch1, data_batch2 = tf.split(data_batch, 2)

    # Train the critic
    with tf.GradientTape() as tape:
        c_loss = - minibatch_energy_distance(
            gen_batch1, gen_batch2, data_batch1, data_batch2, critic,
            sink_iter, entropy_reg
        )
    grads = tape.gradient(c_loss, critic.trainable_variables)
    c_optimizer.apply_gradients(zip(grads, critic.trainable_weights))
    return c_loss


def train_generator(generator, critic, g_optimizer, data_batch,
                    batch_size, latent_dim, sink_iter, entropy_reg):
    """
    Training step of the generator.

    Arguments
    ---------
    generator (Generator): model generating images from latent vector
    critic (Critic): model representing images in abstract feature space
    g_optimizer (tf.keras.optimizer): optimizer for the generator
    data_batch (tf.Tensor): batch of batch_size images from data distribution
    batch_size (int): number of image samples in batch, must be even
    latent_dim (int): dimension of the latent space from which to sample
        generator inputs
    sink_iter (int): number of iterations for sinkhorn algorithm
    entropy_reg (float): regularization term in sinkhorn algorithm
    """
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # Separate batch in 2 equal parts
    data_batch1, data_batch2 = tf.split(data_batch, 2)

    # Train the generator
    with tf.GradientTape() as tape:
        # Generate 2 batches of fake images
        gen_batch = generator(random_latent_vectors)
        gen_batch1, gen_batch2 = tf.split(gen_batch, 2)
        # Get generator loss
        g_loss = minibatch_energy_distance(
            gen_batch1, gen_batch2, data_batch1, data_batch2, critic,
            sink_iter, entropy_reg
        )
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    return g_loss, gen_batch


def minibatch_energy_distance(x1, x2, y1, y2, critic=None, sink_iter=10,
                              entropy_reg=1.0):
    """
    Loss function for generator.
    """
    # Compute sinkhorn divergence between each pair of batches
    x1_y1 = sinkhorn_divergence(x1, y1, critic, sink_iter, entropy_reg)
    x1_y2 = sinkhorn_divergence(x1, y2, critic, sink_iter, entropy_reg)
    x2_y1 = sinkhorn_divergence(x2, y1, critic, sink_iter, entropy_reg)
    x2_y2 = sinkhorn_divergence(x2, y2, critic, sink_iter, entropy_reg)
    x1_x2 = sinkhorn_divergence(x1, x2, critic, sink_iter, entropy_reg)
    y1_y2 = sinkhorn_divergence(y1, y2, critic, sink_iter, entropy_reg)
    # Compute loss
    loss = x1_y1 + x1_y2 + x2_y1 + x2_y2 - 2*x1_x2 - 2*y1_y2
    return loss


@tf.function  # execute in graph mode for acceleration
def sinkhorn_divergence(x, y, critic, sink_iter, entropy_reg):
    batch_size = x.shape[0]
    if critic is None:
        x, y = tf.reshape(x, (batch_size, -1)), tf.reshape(y, (batch_size, -1))
    else:
        # Project images in critic feature space
        x, y = critic(x), critic(y)

    # compute pairwise distance between batches
    cost = cosine_distance(x, y)

    # Apply Sinkhorn algorithm as described in arXiv.1706.00292
    kernel = tf.exp(- cost / entropy_reg)
    v = tf.ones((batch_size, 1), dtype=tf.float32)
    ones = tf.ones((batch_size, 1), dtype=tf.float32) / batch_size
    for i in range(sink_iter):
        u = ones / tf.matmul(kernel, v)
        v = ones / tf.matmul(tf.transpose(kernel), u)
    res = tf.matmul(tf.transpose(tf.matmul(kernel * cost, v)), u)
    return res


def cosine_distance(x, y):
    x_norm = tf.norm(x, axis=1, keepdims=True)
    y_norm = tf.norm(y, axis=1, keepdims=True)
    x, y = x / x_norm, y / y_norm
    cos_dist = 1 - tf.matmul(x, tf.transpose(y))
    return cos_dist


def mnist_preprocessing():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize pixels between 0 and 1
    x_train = x_train.astype('float32') / 255.0  # / 127.5 - 1
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
    return dataset_train


def plot_grid(images, size=8, title=None, start_idx=0):
    """
    Plot a grid to visualize 8*8 images
    """
    # Plot figures
    fig = plt.figure(figsize=(size, size))
    fig.suptitle(title)
    gs = gridspec.GridSpec(size, size)
    for idx in range(start_idx, start_idx + size ** 2):
        ax = plt.subplot(gs[idx - start_idx])
        ax.imshow(images[idx], cmap='gray')
        ax.set_axis_off()
    plt.show()
    plt.close()


def show_hyperparameters(Exp_dir):
    print('Hyperparameters:')
    with open(os.path.join(Exp_dir,'Experiment_info.txt'),'r') as f:
        line = True
        while line:
            line = f.readline()
            print(line, end='')


if __name__ == "__main__":
    dataset_train = mnist_preprocessing()

    Exp_dir = 'Exp_1'
    epochs = 40
    batch_size = 128
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.001
    c_learn_rate = 0.001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_2'
    epochs = 40
    batch_size = 128
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.001
    c_learn_rate = 0.001
    sink_iter = 10
    entropy_reg = 1.0
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_3'
    epochs = 40
    batch_size = 32
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.001
    c_learn_rate = 0.001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_4'
    epochs = 100
    batch_size = 512
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.001
    c_learn_rate = 0.001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_5'
    epochs = 40
    batch_size = 128
    n_gen_per_critic = 6
    latent_dim = 128
    g_learn_rate = 0.001
    c_learn_rate = 0.001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_6'
    epochs = 40
    batch_size = 128
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.001
    c_learn_rate = 0.001
    sink_iter = 100
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_7'
    epochs = 40
    batch_size = 128
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.001
    c_learn_rate = 0.0001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_8'
    epochs = 40
    batch_size = 128
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.0001
    c_learn_rate = 0.001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_9'
    epochs = 100
    batch_size = 128
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.0001
    c_learn_rate = 0.0001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_10'
    epochs = 100
    batch_size = 128
    n_gen_per_critic = 3
    latent_dim = 32
    g_learn_rate = 0.001
    c_learn_rate = 0.001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = True
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)

    Exp_dir = 'Exp_11'
    epochs = 40
    batch_size = 128
    n_gen_per_critic = 3
    latent_dim = 128
    g_learn_rate = 0.001
    c_learn_rate = 0.001
    sink_iter = 10
    entropy_reg = 0.1
    use_critic = False
    training_otgan(dataset_train, epochs, batch_size=batch_size,
                   n_gen_per_critic=n_gen_per_critic, latent_dim=latent_dim,
                   g_learn_rate=g_learn_rate, c_learn_rate=c_learn_rate,
                   sink_iter=sink_iter, entropy_reg=entropy_reg,
                   directory=Exp_dir, use_critic=use_critic)
