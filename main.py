import os
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import imutil
import random
import time

# NOTE: stick this in the parent directory so it's available to multiple runs
FILENAME = '../karras2018iclr-celebahq-1024x1024.pkl'
os.system('wget -nc https://downloads.deeplearninggroup.com/{0} -O {0}'.format(FILENAME))

# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open(FILENAME, 'rb') as file:
    G, D, Gs = pickle.load(file)


def generate_images(latents):
    print('Generating batch of {} images'.format(len(latents)))
    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

    # Run the generator to produce a set of images.
    images = Gs.run(latents, labels)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    return images

# Set max_batch_size to fit your GPU memory constraints
def generate_images_batched(latents, max_batch_size=16):
    i = 0
    while i < len(latents):
        for img in generate_images(latents[i:i+max_batch_size]):
            yield img
        i += max_batch_size


# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
images = generate_images(latents)
# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.jpg' % idx)


# Generate latent videos
latent_start = random.choice(latents)
latent_end = random.choice(latents)
FRAMES = 120
latent_interp = []
for i in range(FRAMES):
    steepness = 6
    gamma = (i / FRAMES) - 0.5
    theta = 1 / (1 + np.exp(-gamma * steepness))
    latent_interp.append(theta * latent_start + (1 - theta) * latent_end)

vid = imutil.VideoLoop('interpolated_face_{}.mp4'.format(int(time.time())))
for img in generate_images_batched(np.array(latent_interp)):
    vid.write_frame(img)
imutil.show(img, filename="face_result.jpg")
vid.finish()
