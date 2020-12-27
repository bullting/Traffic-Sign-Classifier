# Traffic-Sign-Classifier
## Deep Learning Project: Build a Traffic Sign Recognition Classifier
### Project: Traffic Sign Classifier


####Summary of the data set. 

As shown below, I used the numpy and pandas library to calculate summary statistics of the traffic signs data set:

The size of training set is 34,799 images.
The size of the validation set is 4410 images.
The size of test set is 12,630 images
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43.

An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below is sample of image for each class.
 
The bar chart shown the number of training images for each class.
 
	Design and Test a Model Architecture

1.	Preprocessing image.

-	First, I have preprocessed the images by converting to gray scale because as I found from resource on the Internet, the result of using gray scale image is given better performance than RGB. Gray scale level also has less complexity and clear texture and edge.
-	Second, after done some research on the web, there are many kinds of augmenting methods. I used cv2 to transform and warpAffine function to augmented training dataset which generate different edge in each the image and then appended to original training dataset. To have more additional training dataset to train would improve validation accuracy rate (The code shown in cell 5).
-	At last, I normalized the data which would help to process faster.
	The size of new training data set is 112325 images.
-	Below is the sample of preprocessed images:
 

The bar chart below illustrated the number of new data set after adding augmented images.
 
	Model architecture

Below are the details of structure of the convolution network for each layer.

-	Layer1: 
	Input: 32x32x1 image.
	Convolution 5x5: 1x1 stride, valid padding, output = 28x28x6
	Relu activation
	Max pooling: 2x2 stride, output = 14x14x6
-	Layer2:
	Input: 14x114x6 image.
	Convolution 5x5: 1x1 stride, valid padding, output = 10x10x16
	Relu activation
	Max pooling: 2x2 stride, output = 5x5x16
-	Layer3:
	Fully Connected: 400, 120 
	Relu activation
	Dropout
-	Layer4:
	Fully Connected: 120, 100 
	Relu activation
	Dropout
-	Layer5:
	Fully Connected: 100, 43 (Logits)
First, I have implemented the model architecture same as LeNet lab without modifying. With this model, the result of validation accuracy rate was about 91.2%. But, after preprocessed with augmenting data set, the validation accuracy rate increased to 99.7%, however, test accuracy rate was 91.8%

Then, after I applied the dropout at each fully connected layer as I found on University of Cornell web site describes about regularize method using dropout which reduce the problem with overfitting that may probably occurs in this case (The code shown in cell 8). Result of test accuracy rate was improved to from 91.8% to 92.4%.

My final model results were:
	* training set accuracy of 100%
	* validation set accuracy of 99.3% 
	* test set accuracy of 94.2%

	Train model

I used Adam optimizer function for back propagation. The following parameters were used to train the model:

	Batch size = 128
	No. of Epoch = 30
	Learning rate = 0.00097 which generate the best result of validation accuracy rate.
	Dropout = 0.5

After running until Epoch#23, the validation accuracy rate stays around 99.2% to 99.4% which is a maximum rate.

	Test a Model on New Images

1.  Below are five German traffic signs I found on the web:
 

2.	Here are the results of the prediction of each image:

 
All new traffic images were correctly predicted. The result of accuracy turned out to 100%. The code of prediction is in cell 26

3.	Softmax probability

The top of softmax probability of each image is 1.0 as shown below:

INFO:tensorflow:Restoring parameters from ./cv2-g-nd
TopKV2(values=array([[ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.]], dtype=float32), indices=array([[12,  0,  1,  2,  3],
       [22,  0,  1,  2,  3],
       [25,  0,  1,  2,  3],
       [13,  0,  1,  2,  3],
       [34,  0,  1,  2,  3]]))


