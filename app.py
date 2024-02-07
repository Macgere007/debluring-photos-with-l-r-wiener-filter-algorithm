from flask import Flask, render_template, request
import cv2
import numpy as np
from scipy.signal import fftconvolve
from scipy import fft
import os
import base64

app = Flask(__name__)

small_constant = 1e-6

def wiener_filter_gray(blurred, kernel, noise_var):
    # Calculate the Fourier transform of the blurred image
    F_blurred = fft.fft2(blurred)
    
    # Calculate the Fourier transform of the kernel and pad it to match the shape of F_blurred
    F_kernel = fft.fft2(kernel, s=F_blurred.shape)
    
    # Apply the Wiener filter
    F_restored = np.conj(F_kernel) / (np.abs(F_kernel)**2 + noise_var)
    restored = np.real(fft.ifft2(F_blurred * F_restored))
    
    return np.clip(restored, 0, 255)

def lucy_richardson_gray(image, kernel, iterations):

    kernel /= np.sum(kernel)
    
    for _ in range(iterations):
        est_blur = fftconvolve(image, kernel, 'same')
        ratio = image / est_blur
        image *= fftconvolve(ratio, kernel[::-1, ::-1], 'same')
    
    return np.clip(image, 0, 255)

def wiener_filter_color(blurred, kernel, noise_var):
    # Separate channels for color images
    channels = [fft.fft2(channel) for channel in cv2.split(blurred)]
    
    # Apply Wiener filter to each channel
    restored_channels = []
    for channel in channels:
        F_kernel = fft.fft2(kernel, s=channel.shape)
        F_restored = np.conj(F_kernel) / (np.abs(F_kernel)**2 + noise_var)
        restored_channel = np.real(fft.ifft2(channel * F_restored))
        restored_channels.append(restored_channel)
    
    # Merge the restored channels
    restored = cv2.merge(restored_channels)
    return np.clip(np.abs(restored), 0, 255).astype(np.uint8)

def lucy_richardson_color(image, kernel, iterations):

    kernel /= np.sum(kernel)

    # Separate channels for color images
    channels = [image[..., i].astype(float) for i in range(image.shape[-1])]

    # Apply Lucy-Richardson to each channel
    restored_channels = []
    for channel in channels:
        for _ in range(iterations):
            est_blur = fftconvolve(channel, kernel, 'same')
            ratio = channel / (est_blur + small_constant)  # Use small_constant to avoid division by zero
            channel *= fftconvolve(ratio, kernel[::-1, ::-1], 'same')
        
        restored_channels.append(np.clip(channel, 0, 255))

    # Merge the restored channels
    restored = np.stack(restored_channels, axis=-1)
    return np.clip(restored, 0, 255).astype(np.uint8)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        # if the file format isn't supported
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        uploaded_file = request.files['file']

        # if the user does not select a file
        if uploaded_file.filename == '':
            return render_template('index.html', error='No selected file')

        file_path = os.path.join(os.path.dirname(__file__), 'uploaded_image.jpg')

        uploaded_file.save(file_path)


        # Define kernel and noise variance to adjust blur reduction
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        noise_variance = 0.05
        # blur
        blurred_image = cv2.imread(file_path)
        #perform image deblurring
        wiener_restored_color = wiener_filter_color(blurred_image, kernel, noise_variance)
        lucy_richardson_restored_color = lucy_richardson_color(wiener_restored_color, kernel, iterations=10)
        blend_ratio = 0.5
        combination_color = np.average([wiener_restored_color, lucy_richardson_restored_color], axis=0, weights=[blend_ratio, 1 - blend_ratio])
        combination_color_2= np.clip(combination_color, 0, 255)

        #image encoder
        blurred_image_color_base64=base64.b64encode(cv2.imencode('.png', blurred_image)[1]).decode('utf-8')
        wiener_color_base64 = base64.b64encode(cv2.imencode('.png', wiener_restored_color)[1]).decode('utf-8')
        lr_color_base64 = base64.b64encode(cv2.imencode('.png', lucy_richardson_restored_color)[1]).decode('utf-8')
        combination_color_base64 = base64.b64encode(cv2.imencode('.png', combination_color_2)[1]).decode('utf-8')
    
        return render_template('result.html',
                            blur_color=blurred_image_color_base64,
                            wiener_color=wiener_color_base64,
                            lr_color=lr_color_base64,
                            combination_color=combination_color_base64,
                            uploaded_image_path=file_path)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
