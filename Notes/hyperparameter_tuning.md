# Hyper-Parameter Tuning
## Week 1
### Splitting Data
<ul>Hyperparameters: <li>No of layers</li> <li>No of hidden_units</li> <li>learning_rate</li> <li>activation_function</li> <li>regularization_parameter</li></ul>
<br>

<p>data splitting: train/dev/test
<br>previously: 60 /20 /20
<br>current : depending on size data: eg 98/1/1</p>

<br>

<p>High Variance = Not able to generalise to test set
<br>high bias = Does not fit to train set well</p>
<br>
<table>
<th>train_set_error</th>
<th>test_set_error</th>
<th>Conclusion</th>
<tr><td>Low</td>
<td>High</td>
<td>High Variance</td>
</tr>
<tr><td>High</td>
<td>Low</td>
<td>High Bias</td>
</tr>
<tr><td>High</td>
<td>High</td>
<td>High Variance and High Bias</td>
</tr>
</table>
<br>
<ul>Algorithm:
<li>(High bias?) :Y => bigger network, Train longer, try changing NN architecture -->(High bias?)</li>
<li>(High bias?) :N => 
<ul>
<li>(High variance?) :Y => More data,regularization, try to change NN architecture  --> (High bias?) </li>
<li>(High variance?) :N =>Done</li>
</ul>
</li>

</ul>
<br>

### Regularization :
(or **weight decay**) 
#### L2 regularization:
<p>
Logistic: 

``` python
regularization term = (lambd/(2*m))*||w||^2 = (lambd/(2*m))*((w.T).w)
```
<br> NN : 
``` python
(lambd/(2*m))*sum(sum(sum(W^2(i,j,l))))
```
</p>
<br>
Update parameters

``` python
W[l] := W[l](1-alpha/m) - (alpha)*dW[l]
```
<br>

#### Dropout regularization: 
<p>
(Used most commonly for computer vision)
This helps to make the NN architecture smaller and thereby helps to avoid overfitting. By not relying on only one feature, the weights are spread out across the layers. This causes shrinking of the weights which similar to L2 norm. We can have different keep_prob values for different layers. We can also have a keep_prob for the input layer(not suggested).
</p>
<p>When using dropout,we have to multiply by keep_prob during test time.</p>
<br>
Disadvantages :
<p>Cost function is not well defined. Since we are randomly killing of nodes it is difficult to cross check during gradient descent. One way to overcome this is to have keep_prob = 1 and check gradient descent. And we have to hope that we did not create any errors during implementation of dropout.
</p>
<br>

#### Inverted Dropout Technique:
<p>for layer l:

```python
d[l] = np.random.rand(a[l].shape[0],a[l].shape[1]) < keep_prob  # this will give a boolean array of True(or 1) and False(or 0). 
a[l] = np.multiply(a[l],d[l])     # this will shut off the neurons where the rand no generated is less than keep_prob
a[l] /= keep_prob    # In order to not reduce the value tof the the next layer ie a[l+1]. 
```
<br>
<br>The same d[l] is used for both forward prop and backprop of the same itereation. For the next iteration a new d[l] is generated
<br>During test time we do not use dropout. This is used to avoid the noise in the output caused due to zeroing out the neurons. We can also use dropout during the test time but we would have to take the average of the output we get by iterating through the test set multiple times.
</p>
<br> 


#### Data Augmentation:
<p>Increasing the data by distorting the data set. This helps to decrease overfitting. </p>

#### Early Stopping:
<p>
Plot dev set error and trainset error while training. Then stop the algorithm when the dev set error starts to increase after reaching the minima.
<br>
Disadvantage: it tries to couple minimising cost and decreasing overfitting together. Ideally we would try to handle the two problems seperately.
<br>
Advantage: We need not try many values of lambda to choose the best vale of regularisation parameter, which is computationally expensive.</p>

<br>
<br>

### Useful Practices for Optimization:
#### Normalization:
<p>Normalization is required when we have data of different scales  

```python
X := X-mu
X := X/(sigma)
```
<br>
Use the same mu and sigma for both train set and test set
</p>
Advantages:

* Speeds up training
* For un-normalized data, we must use smaller learning rate thus it will take longer to converge.

#### Vanishing/Exploding Gradients:
<p>If we have a very deep network, when we calculate the gradients for back prop, from chain rule, we get the gradients to be the multiplication of all the derivative terms from the layers ahead. If there are many gradients smaller than 1 then the product becomes very small and in turn a very small update(after multiplying with a small learning rate) is done to the weight. Thus there is no significant change to the weights and thus the gradient descent does not learn anything. Similarly, gradient explodes when the values are greater than 1 and thus causing gradient descent to not converge.</p>

<p>To rectify it:
for 1 neuron: z = w1x1+w2x2+......+wnxn +b
large n => z is large =>  we want w_i to be smaller.
We would want to set the 
Initialization: 

* For Relu => 
``` python
W[l] = np.random.randn(shape) * np.sqrt(2/n[l-1])
```
* For tanh => 
``` python
W[l] = np.random.randn(shape) * np.sqrt(1/n[l-1])  (Xavier's initialisation)
```
* For tanh =>
``` python
W[l] = np.random.randn(shape) * np.sqrt(1/(n[l-1]+n[l]))  (Yoshua Bengio's initialisation)
```
</p>

#### Gradient Checking:
<p>To check if back prop is working correctly
Take W[1],b[1],W[2],b[2],......W[L],b[L] and reshape it to a vector Theta. Similarly the derivatives(dW[1],..db[L]) to dTheta.
<br>

``` python
dTheta_approx[i] = (J(theta[1],theta[2],...theta[i]+epsilon,...)-J(theta[1],theta[2],...theta[i]-epsilon,...))/(2*epsilon)
```
Now compute error:
``` python
 error = (||dTheta_approx-dTheta||)/(||dTheta_approx||+||dTheta||)
```
<ul>Using epsilon = 10^-7 :
<li>if error ~ 10^-7  ---- Working good</li>
<li>if error ~ 10^-5  ---- May be ok, Check for bugs</li>
<li>if error ~ 10^-3  ---- Check for bugs, by comparing dTheta and dTheta_approx</li>
</ul>
</p>
<br>

## Week 2:
### Mini-Batch Gradient Descent: 
<p>When we have a large data, if we update our parameters at the end of after going through the whole data set, it would take a lot of time for the cost to converge to a smaller value. To rectify this we use mini batches. We update the parameters after a number of data, for eg 1000. This will help the algorithm to  let the cost onverge to a minima faster and in lesser number of iterations.
</p>
<br>
<p>
Mini-batch size is also a hyperparameter.
<br>
Batch Gradient Descent: Takes too long per iteration.
<br>
Stochastic Gradient Descent: We lose the advantage provided by vectorization for the number of examples.
<br>
Mini-Batch Gradient Descent: Quicker updates and we do not lose vectorization.
</p>
<p>
Usually we keep min-batch size as powers of 2, since the advanced libraries like tensorflow and pytorch provide better performance.
</p>

### Exponentially Weighted Averages
``` python
v[t] = beta * v[t-1] + (1-beta)*Theta[t]
```
<p>This is approximately same as the weighted average of the last 1/(1-beta) days.</p>
<p>Using Bias correction:</p>

```python
v[t] = beta * v[t-1] + (1-beta)*Theta[t]
v[t] = v[t]/(1-(beta)^t)
```
<p>Update Rule:</p>

```python
VdW = beta*VdW + (1-beta)dW
Vdb = beta*Vdb + (1-beta)db
VdW = VdW/(1-(beta)^t)
Vdb = Vdb/(1-(beta)^t)
W = W - alpha * VdW
b = b - alpha * Vdb
```
### RMSprop
<p>Another way to find exponentially weighted averages.</p>

```python
sdW = beta2*sdW + (1-beta2)dW^2
sdb = beta2*sdb + (1-beta2)db^2
W = W - alpha * dW/((sdW)^0.5 +epsilon) # epsilon = 10^(-8)
b = b - alpha * db/((sdb)^0.5 +epsilon) # epsilon = 10^(-8)
```
### Adam Optimization Algorithm:
<p>Also known as Adaptive Moments Estimation</p>

```python
VdW = beta1*VdW + (1-beta1)dW    # First moment
Vdb = beta1*Vdb + (1-beta1)db
sdW = beta2*sdW + (1-beta2)dW^2  # Second moment
sdb = beta2*sdb + (1-beta2)db^2
VdW = VdW/(1-(beta1)^t)
Vdb = Vdb/(1-(beta1)^t)
sdW = sdW/(1-(beta2)^t)
sdb = sdb/(1-(beta2)^t)
W = W - alpha * VdW/((sdW)^0.5 +epsilon) # epsilon = 10^(-8)
b = b - alpha * Vdb/((sdb)^0.5 +epsilon) # epsilon = 10^(-8)
```
<ul>Hyperparameters:
	<li>alpha = Needs Tuning</li>
	<li>beta1 = 0.9</li>
	<li>beta2 = 0.999</li>
	<li>epsilon = 10^(-8)</li>
</ul>

### Learning Rate Decay:
<p>Different ways to decay alpha</p>

* alpha = alpha0/(1+decay_rate*epoch_num)
* alpha = (0.95^epoch_num)*alpha0
* alpha = alpha0*k/epoch_num^0.5   #K is another hyperparameter
* alpha = alpha0*k/mini_batch_no^0.5 
* alpha takes discrete values, for certain part of the training, then reduced for another set. And it keeps reducing for discrete values.
* Manually decreasing values

<p>Since there are a lot of parameters in a neural network, the chance of reaching</p>

## Week 3:
### Order of tuning:
<ol type="1">
	<li>Learning Rate</li>
	<li><ul>
		<li>beta (if using exponentially weighted averages)</li>
		<li>No of hidden units</li>
		<li>learning rate decay</li>
	</ul></li>
	<li><ul>
		<li>No of layers</li>
		<li>learning rate decay</li>
	</ul></li>
	<li>Need not tune: the hyperparameters of Adam optimisation, ie beta1 = 0.9,beta2 = 0.999, epsilon = 10^-8</li>
</ol>

### How to Tune:
* Instead of using a grid, sample the hyperparameters randomly in a range.
* Use a coarse to fine search process, ie sample points more densely closer to the region which gives least error.
* For hyperparameters like number of layers/no of units, we can just sample random natural numbers on the number line.
* For hyperparameters like learning rate or beta(if using exponentially weighted averages), since it is more sensitive at some regions than other it is better to sample randomly on the logarithmic scale.
``` python
# for learning rate, between 0.0001 and 1,
r = -4*np.random.rand()
alpha = 10**r
```
<br>
<p>Two different approaches to tuning hyperpameters</p>

* Panda approach:
	1. Used when we do not have much computational resources and thus we can only train one model at a time.
	2. Watch the learning curve and adjust parameters manually and keep changing the parameters each day when it is train for many days.
* Caviar approach:
	1. Used when we do have computational resources to train many models at a time.
	2. Run many models at the same time and choose the one which will give the least error.

### Batch Normalization:
* In a NN, we can normalize before taking the activation or even after it. Normalizing before taking activation is more common.
* mu = mean(Z[l])    # Z[l] is l'th layer of the NN(before taking activation), mean taken about all m examples(may be mini-batch).
* sigma = (var(Z[l]))^0.5
* Z_norm[l] = (Z-mu)/(sigma^2 + epsilon)^0.5
* Z_new = gamma * Z_norm + beta    # gamma and beta are parameters for layer l.  
* Update the parameters as beta = beta - learning_rate * dbeta          # or any other optimization algorithm
* **Now when we compute mean, b[l] irrespective of the value gets cancelled out. Thus we do not need b[l] parameters wherever we perform batch normalization.**

#### Why does Batch Norm Works:
* Normalization speeds up the network. Similarly batch normalization does it on all layers.
* It makes weights deeper in the NN more robust to changes in initial layers.
* If the NN is trained to classify black cats, then if we change the inputs from black cats to coloured cats, then the network would not be able to classify them due to a covariance shift.
* Batch norm helps one layer to learn more independent of the layers before it. Since the mean and variance does not change due to the previous layers' weights. Hence there is a lesser "covariance shift" than before owing to a better performance.
* Since the batch norm takes the mean and var on the mini-batches, there is a slight noise added to it. This noise propagates forward and it also causes a similar effect like dropout but it is less significant.

#### Batch Norm at test time:
* During training take an exponentially weighted average of the mean and variance. Then use the latest values of the mean and variance for the test set.

### Softmax Regression: 
* Softmax activation funtion -> A = Z/sum(exp(Z)) # sum is over all the neurons. 
* Softmax is used only for the last layer.
* Softmax can be used for C classes. Also for  C = 2, it gives logistic regression.
* Loss function = - sum(y * log(y_predicted))


## Guidelines:
<ul>
<li> <b>Make sure that the dev and test sets come from same distribution.</b> </li>
<li>If we don't need an unbiased estimate of our algorithm's performance then it is okay to tune hyperparameters on the test set</li>
<li>When implementing regularization, to debug gradient descent, plot cost function(including the regularization term) with respect to number of iterations and check if it is decreasing monotonically. </li>
<li>Run Gradient Checking algorithm only when debugging and do not keep it running while training data. </li>
<li>If grad check error is high then look at components to try to look at the components which have higher error and see if there is any pattern and debug that.</li>
<li>For grad check do not forget to include regularization in the derivative terms.</li>
<li>Grad check doesn't work with drop out. For checking error while using dropout, take keep_prob = 1 and then implement grad check</li>
<li>It might happen that gradient descent is correct when the parameters are close to 0 but the errors increase when the values deviate from 0. Then let the NN train for sometime to let the parameters deviate from 0 and then apply grad check.</li>

</ul>
<br>

## Programming Assignment Results:
<ul>Week 1:
<ul>Initialization:
<li>Initialization with zeros fails to break symmetry.</li>
<li>Also initialization with zeros causes Z1 = 0. If we are using Relu for activation function, it implies that a1 = 0(not the case when we are using sigmoid, for sigmoid a1 = 0.5). This causes all the layers ahead to have a value of 0 (except last layer since we use sigmoid for that). Now, during back prop, dZ3 = a3 - y. This has a non zero value. But when we are calculating dW /dZ2, we have to multiply dZ3 with a2.T or W3.T both of these are 0. Thus nothing is being updated to these. Also in their data set, sum of dz3 was coming out to be close to 0, this caused the NN to not learn in the bias terms as well. Thus the cost remained same.</li>
<li>When we randomly initialize the parameters to random but big values the algorithm learns slowly, and thus will require more number of iterations over the training set to reach the optimum value. </li>
<li>He initialization is a good choice for ReLU: initialize with factor of np.sqrt(2/previous_layer_dim)</li>
</ul>
<br>
<ul>Regularization:
	<li>With no regularization the training set error is least but the test set error is the most.</li>
	<li>Using regularization either dropout or L2 norm, increases the training set error but it reduces the test set error.</li>
</ul>
<br>
<ul>Gradient_Checking:
	<li>

``` python
dTheta_approx[i] = (J(theta[1],theta[2],...theta[i]+epsilon,...)-J(theta[1],theta[2],...theta[i]-epsilon,...))/(2*epsilon)
```
</li>
	<li>The above calculated value is close enough to the values calculated during back-prop. Thus we can use gradient checking to verify whether our back-prop is correct or not</li>
	<li>While finding gradient approx make a copy of the parameters so as to retain the original parameters. Python will make changes to all the variables which have been created not as a copy.</li>
</ul>
</ul>
