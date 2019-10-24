# MachineLearning
Machine Learning exercises study year 19-20

# Final Exercise

##[ ]1- Multi Layer Perceptron (MLP).

Modify the provided script 'perceptron.py' to build a MLP. Use architectures 
with 0, 1 and 2 hidden layers. Keep the complexity of the model bounded so runs do not take much more
than 1 hour to reach the maximum of testing accuracy. Notice that the input needs to be "flattened" since there is no spatial structure 
in this fully connected design.  This can be achieved by adding a dummy layer with no free parameters with "layers.Flatten()"
as the first layer in the constructor "model.Sequential()". Obtain the learning curves and discuss the results.
Report the optimizer in use, initialization parameters, the learning rate, etc. Is early stopping convenient
in this model?

##[ ]2- Biggest MLP

Reuse the code from part 1 to build and run a MLP with one hidden layer as big a you can. 
Compare the performance of your design with the results appearing in Table 1 of [https://arxiv.org/pdf/1611.03530.pdf] for a MLP of 512 units in a single 
hidden layer. Report the best result found for a maximum of 1000 epochs or 2 hrs CPU running time.
The best accuracy amongst all teams will be awarded extra points.

##[ ]3- Convolutional Networks.

 Study the performance properties of the convolutional network provided in the Tensorflow tutorial. How is 
the learning affected if instead of ReLU units, tanh() activations are used? What is the reason for this? Compare also
at least two different optimizer algorithms.

##[ ]4- Outperforming Convolutional Network

Try to outperform the convolutional network of part 3 with a MLP that uses approximately the same number of parameters.
Report your results and explain them.
