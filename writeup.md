# **Traffic Sign Recognition** 

Here is a link to my [project code](https://github.com/VamshiK-Kasula/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[original_test_data]: ./results/original_test_data.png  "Test images"
[histogram]: ./results/histogram.png  "Histogram"
[test_images]: ./results/new_images.png  "Test Images"
[accuracy]: ./results/accuracy.png  "Accuracy"


### Data Set Summary & Exploration

Data set for the Traffic sign recognition is obtained from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). It contains test validation and training sets.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Following are the unique classes/labels set:

![alt text][original_test_data]

The 34799 data set from the training set is distributed as follows:
![alt text][histogram]


### Pre-process the Data Set

THe given training set is extended by duplicating the available data and rotating or translating the duplicated data. Rotating and translating is done keeping in the view that the images are not captured ideally. Totally the data set is increased to 382789 from 34799 data samples.

```python
def rotate_image(image, angle):
    M = cv2.getRotationMatrix2D((image.shape[0] / 2, image.shape[1] / 2), angle, 1)
    return cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))
```

```python
def translate_image(image):
    t_mat = np.float32([[1,0,np.random.randint(-5,5)],[0,1,np.random.randint(-5,5)]])
    translate_image = cv2.warpAffine(image,t_mat,(32,32))
    return translate_image
```
After extending the given data, the images are added with gaussian blurr and converted into Grayscale.

```python
def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def pre_process(image):
    blur = cv2.GaussianBlur(image, (3,3) ,0)
    blurred_image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    gray= grayscale(blurred_image)
    return np.array(0.8*(gray/255.0)+0.1, dtype=np.float32).reshape(32,32,1)

```

Finally all the images are converted to (32, 32, 1) shape, which is essential in our next steps.

### Design and Test a Model Architecture

The final model architecture is adapted as described in the course lecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image 						| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 10x10x16 				|
| Flatten	outputs	    | 400 outputs    								|
| Fully connected		| 120 outputs      								|
| RELU					|												|
| Fully connected		| 84 outputs        				     		|
| RELU					|												|
| Fully connected		| outputs = number of lanels    	     		|



To train the model, I used 25 epochs with a batch size of 256.

My final model results were:
* training set accuracy of 0.904
* validation set accuracy of 0.892 
* test set accuracy of 0.907

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

The following new images are tested using the model

![alt text][test_images]

Here are the results of the prediction:

| Image			                                        |     Prediction	                   					| 
|:-----------------------------------------------------:|:-----------------------------------------------------:| 
| Go straight or rightStop Sign      		            | Go straight or rightStop Sign                         |
| Road workU-turn     			                        | Road workU-turn     			                        |
| No vehiclesYield					                    | No vehiclesYield					                    |
| No passing for vehicles over 3.5 metric tons100 km/h	| No passing for vehicles over 3.5 metric tons100 km/h  |
| Speed limit (60km/h)Slippery Road			            | Speed limit (60km/h)Slippery                          |
| Turn right ahead                                      | Turn right ahead                                      |
|  <span style="color:red">Roundabout mandatory </span> |  <span style="color:red">Priority road      </span>   |
| Speed limit (30km/h)                                  | Speed limit (30km/h)                                  |
| General caution                                       | General caution                                       |
| Road narrows on the right                             | Road narrows on the right                             |



The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. Roundabout mandatory sign was classified wrongly as Priority road

the prediction breakdown of each images are as follows:
---
### Results
| Road Sign             | Prediction    | 
|:---------------------:|:-------------:| 
|<img src=./results/36.png width="100"> |**36 Go straight or right:**<br> 36: 90.25%<br> 38: 2.98%<br> 40: 1.39%<br> |
|<img src=./results/25.png width="100"> |**25 Road work:**<br> 25: 96.19%<br> 29: 0.63%<br> 38: 0.58%<br>|
|<img src=./results/15.png width="100"> |**15 No vehicles:**<br> 15: 95.94%<br> 12: 0.95%<br> 38: 0.75%<br>|
|<img src=./results/10.png width="100"> |**10 No passing for vehicles over 3.5 metric tons:**<br> 10: 97.51%<br> 5: 1.64%<br> 2: 0.43%<br>|
|<img src=./results/3.png width="100"> |**3 Speed limit (60km/h):**<br> 3: 99.66%<br> 5: 0.34%<br> 2: 0.00%<br>|
|<img src=./results/33.png width="100"> |**33 Turn right ahead:**<br> 33: 99.89%<br> 37: 0.04%<br> 35: 0.04%<br>|
|<img src=./results/12.png width="100"> | <span style="color:red">**12 Priority road:**<br> 12: 47.23%<br> 42: 8.82%<br> 40: 7.94%<br> </span>|
|<img src=./results/1.png width="100"> |**1 Speed limit (30km/h):**<br> 1: 90.92%<br> 0: 8.00%<br> 2: 0.58%<br>|
|<img src=./results/18.png width="100"> |**18 General caution:**<br> 18: 97.90%<br> 26: 1.85%<br> 27: 0.25%<br>|
|<img src=./results/24.png width="100"> |**24 Road narrows on the right:**<br> 24: 79.32%<br> 18: 7.96%<br> 27: 5.33%<br>|
