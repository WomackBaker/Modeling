import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras import initializers

csv = pd.read_csv('../outputcp.csv')
data = csv.values

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

# Generator
def build_generator(latent_dim, output_dim):
    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim, kernel_initializer=initializers.RandomNormal(stddev=0.03)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='tanh'))
    return model

# Discriminator
def build_discriminator(input_dim):
    model = Sequential()

    model.add(Dense(512, input_dim=input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.03)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile the models
latent_dim = 1
output_dim = data.shape[1]

discriminator = build_discriminator(output_dim)

discriminator.load_weights('discriminator_weights.h5')
discriminator.compile(loss=['binary_crossentropy', 'mean_squared_error'], optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim, output_dim)
generator.load_weights('generator_weights.h5')
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

def train_gan(epochs=10000, batch_size=128):
    for epoch in range(epochs):
        # Generate a random batch of sequences
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_sequences = generator.predict(noise) * 1  # Scale to be between 0 and 100

        # Select a random batch of real sequences
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_sequences = data[idx]

        # Labels for generated and real data
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_sequences, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_sequences, labels_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        labels_gan = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, labels_gan)
        if epoch % batch_size == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            samples=100
            noise = np.random.normal(0, 1, (samples, latent_dim))
            generated_sequences = generator.predict(noise) 
            generated_sequences = generated_sequences.reshape(samples, -1) 
            idx = np.random.randint(0, data.shape[0], samples)
            real_sequences = data[idx]
    generated_sequences = scaler.inverse_transform(generated_sequences)
    plot_comparison(generated_sequences[0], real_sequences[0], epochs)
    with open('Daccuracy.txt', 'w') as f:
            f.write('D accuracy: ' + str(d_loss[1]))

    generated_sequences = scaler.inverse_transform(generated_sequences)
    plot_comparison(generated_sequences[0], real_sequences[0], epochs)

def plot_comparison(generated_data, real_data, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(generated_data, label='Generated Data', marker='o', linestyle='-', alpha=0.7)
    plt.plot(real_data, label='Real Data', marker='o', linestyle='-', alpha=0.7)
    plt.title(f'Generated vs Real Data Comparison (Epoch {epoch})')
    plt.xlabel('Sequence Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('ganout.png')

while True:
    train_gan()
    generator.save_weights('generator_weights.h5')
    discriminator.save_weights('discriminator_weights.h5')