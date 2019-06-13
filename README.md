# TAKING-IDENTITY-INFORMATION-WITH-CAMERA-BY-USING-TENSORFLOW-AND-OPENCV
Hi everybody!

Me and my buddy @krsad decided to develop a project about taking identity information from ID cards.

We used Tensorflow and OpenCV for detecting ID cards. Also we used Tesseract OCR for detecting and taking the texts on the cards.

We coded in python and run it on Ubuntu.

Generally, we runned our program in real time. So we arranged the codes to run with a smartphone. We did this, because a camera of smartphone is better than a webcam. However, it is possible to study on a photograph or on a video. Also those files could be find in this tutorial.

We still working on this project and keep developing it.

<h2>Installation</h2> 

Running Tesseract on Windows is not easy, so we used Ubuntu as operating system.

We trained our model for detecting 3 ID cards which are Rebuplic of Turkey Identity Card, Driving Licence and University of Ankara Identity Card. You can directly use our trained model or you can train your own model. So, you have 2 options:

1- You can train your own model and you can use it on your own project. It has already been shown in <a href=https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10>here</a>. Or,

2- You can use the model that we trained for this project (to be able to run this project on your computer). This is going to be shown in this tutorial.

Install Python3 (we used 3.6) and PyCharm could be used as IDE.

Note: If you currently use Tensorflow for your some another projects, setting up a virtual environment and using it for this project would be better for you to prevent probable problems. We recommend Anaconda Virtual Environment.

For Tensorflow, we have 2 options; Tensorflow CPU and Tensorflow GPU. If you have GPU Driver, we highly recommend you to use Tensorflow GPU.

<h2>Tensorflow</h2>
<h3>1. Tensorflow CPU</h3>

Open command line and write down this code:
```ruby
C:\> pip install --upgrade tensorflow
```
That's all for CPU. You can test it by opening python terminal in the command line. Write down "python". Now, write the codes below:
```ruby
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```
If you take "Hello, Tensorflow!" as output, it means that everything is OK.

<h3>2. Tensorflow GPU</h3>

Firstly, CUDA and CUDNN which are libraries of NVidia, have to be installed. Also don't forget to update your GPU driver to latest version.

Download CUDA version 10.0 from its website and install it. After installation, new paths have to be added in "environment variables" in your computer just as below. However, if you installed CUDA in an another directory, make necessary changes on the paths below.

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64
```

Download CuDNN 7.4.2 for CUDA 10.0 from its website. Open "bin" file from downloaded file and copy cudnn64_7.dll. Paste it into the "bin" file of CUDA. Default directory would be like this:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
```
Now, open command line and run this code:

```ruby
C:\> pip install --upgrade tensorflow-gpu
```

You can test it as we did for Tensorflow CPU.

Also run the codes below in command line for installing other necessary libraries.

```ruby
C:\> pip install --ignore-installed --upgrade tensorflow-gpu
C:\> conda install -c anaconda protobuf
C:\> pip install pillow
C:\> pip install lxml
C:\> pip install jupyter
C:\> pip install matplotlib
C:\> pip install pandas
C:\> pip install opencv-python
```
Note: If you installed the Tensorflow for the CPU, delete the "-gpu" from the first line.

<h2>Object Detection</h2>

Firstly, download Tensorflow Object Detection API from <a href=https://github.com/tensorflow/models>here</a>.

Then, download files of this project.

Paste it into the models-master/research and approve the changes.

You can now identify 3 ID cards in webcam. But, it won't work yet, because connection to smartphone have to be done. In addition, Tesseract have to be installed.

<h2>Smartphone Connection and Tesseract OCR</h2>

We used an Android app but finding an iOS app is possible at the App Store. You can find it in <a href="https://play.google.com/store/apps/details?id=com.pas.webcam&hl">here</a>, install it on your smartphone. For connect your smartphone to computer, start the app. Click on "Start Server". Now, you can see the IP address and the port number. Open the "object-detection-tutorial-webcam.py" file in object_detection file. Write down the IP address and the port number into the code. You can easily find the line where they must be written.

Lastly, Tesseract have to be installed for text detection. Download it in <a href=https://github.com/tesseract-ocr/tesseract>here</a> and install it by following the instructions.

Installation is done.

<h2>Using the Programme</h2>

Start the server on your smartphone.

Run "object-detection-tutorial-webcam.py". View of camera of smartphone will be displayed on the computer screen.

Focus on any of the 3 cards (Rebuplic of Turkey Identity Card, Driving Licence and University of Ankara Identity Card) and take a photo by pushing on the "Q" on keyboard when the camera is well-focused.

You can see the detected texts (ID Number and names) as the output of the code.

Also, you will see the stages on the screen which have been runned after taking the photo.

If you train your own model, you can easily arrange all of these codes for your own project.

<h2>Consequences</h2>

Here, you can see the examples from steps of the programme.

1- Detected object(driving licence) can be seen in the photo below:

<img src="https://github.com/kscompsci/TAKING-IDENTITY-INFORMATION-WITH-CAMERA-BY-USING-TENSORFLOW-AND-OPENCV/blob/master/WhatsApp%20Image%202019-04-11%20at%2016.53.16%20(2).jpeg" alt="Photo Detection">

2- You can see the cropped object below:

<img src="https://github.com/kscompsci/TAKING-IDENTITY-INFORMATION-WITH-CAMERA-BY-USING-TENSORFLOW-AND-OPENCV/blob/master/WhatsApp%20Image%202019-04-11%20at%2016.53.16%20(1).jpeg" alt="Photo Clip">

3- Here you can see 3 (perspective transformation, denoising and contrast) stages:

<img src="https://github.com/kscompsci/TAKING-IDENTITY-INFORMATION-WITH-CAMERA-BY-USING-TENSORFLOW-AND-OPENCV/blob/master/WhatsApp%20Image%202019-04-11%20at%2016.53.16.jpeg" alt="Stages of Programme">

4- Finally, taken ID Number(TC Kimlik Numarası) and name can be seen below:

<img src="https://github.com/kscompsci/TAKING-IDENTITY-INFORMATION-WITH-CAMERA-BY-USING-TENSORFLOW-AND-OPENCV/blob/master/WhatsApp%20Image%202019-04-11%20at%2016.53.15.jpeg" alt="Stages of Programme">

<h2>For Errors and Help</h2>

Installing Tensorflow won't be so easy. We highly recommend you to check <a href="https://www.tensorflow.org">tensorflow.org</a> for solving problems which will occur during the installation and more information about Tensorflow can be found there.


5- This term all the codes have been updated because of the development on client. You can easily reach the android application on our github. There is a server side of the project also. We will add that later. https://github.com/kscompsci/id-reader-android-app => here is the link on the application. We are waiting your support to develop the project:)

One of the useful website is, of course, <a href="https://opencv.org">opencv.org</a>. Check it!
