# Deep-Learning-Mini-projects
## Meetings
### 6. April
ToDos discussed:<br>
- Raphael starts GitHub repo [DONE] <br>
- Lena sets up template for report [DONE] <br>
- Marijn sets up image generation for prediction and tests [DONE]<br>
<br>
Project Idea: <br>
After we are all set up, we start with a first model, which we will then use as a baseline to implement more optimized architectures. Once we have the baseline, everyone works on implementing different architectures. To discuss the implementation of the baseline, we will have another meeting on Friday 9th.<br>

### 9. April
Marijn and/or Raphael start implementing Baseline model, then everyone thinks about possible architectures and maybe starts implementing them already. We do a new notebook for each architecture to avoid merge conflicts. We will meet again to talk about the architectures on Friday 16th.
### 16. April 15:00
We identified the following questions, which have been posted on Slack: <br>
- Can we use kmeans for preprocessing? Yes. <br>
- Can we use pretrained models? Yes. <br>
- What do the time and error rate in the project description mean to us? Just a rough guideline. <br>
We plan on proceeding as follows: <br>
Lena: Keep improving the base model and add auxiliary losses <br>
Raphael: Siamese models <br>
Marijn: Convolution (thinning out) <br>
What we also could look at would be Autoencoders and UNet
### 23. April 12:00
We identified the following questions, which have been posted on Slack: <br>
- Can we use more than 25 epochs? Yes. <br>
- Can our total training time be longer than 1 minute? <br>
Lena: write report, linear model, autoencoder <br>
Raphael: improve Siamese model <br>
Marijn: improve RNN <br>
### 30. April 13:00  
Project 1: <br>
Lena: finish report, performance analysis and drawings, hand over to Raphael and Marijn for polishing <br>
Project 2: <br>
Start list with first-come-first-serve tasks, help yourselves ;) <br>
We identified the following questions, which have been posted on Slack: <br>
- Do we have to implement SGD ourselves?
### 7. May 13:00  
Todo:  
- Tanh module (Raphael)
### 14. May 13:30
Todo:
- Sequential
- Test Cases
- ReLu (Lena)
<br>
Done: <br>
- Linear (Raphael)<br>
- Tanh (Raphael)<br>
- MSE (Marijn)<br>
- Training & Test Set (Marijn)<br>

### 21. May 13:00
## Project 1
Scope: 3 weeks <br>
Preferred Deadline: 30th April <br>
Idea for Baseline: activation function: Linear, 1 hidden layer (ReLU), 50 units, fully connected network, 25 epochs, Loss: Cross-Entropy, maybe SGD as optimizes <br>
## Project 2
Scope: 3 weeks <br>
Preferred Deadline: 21st May <br>
Objective: The objective of this project is to design a mini “deep learning framework” using only pytorch’s tensor operations and the standard math library. <br>
**List of Tasks:** <br>
*Note: Your framework should import only torch.empty. Your code should work with autograd globally off, which can be achieved with
torch.set_grad_enabled(False)* <br>
**Linear Module:**<br>
- init <br>
- forward <br>
- backward <br>
- param <br>
**ReLU:**<br> (Lena)
- init <br>
- forward <br>
- backward <br>
- param <br>
**Tanh:**<br> (Raphael) 
- init <br>
- forward <br>
- backward <br>
- param <br>
**Sequential:**<br>
- init <br>
- forward <br>
- backward <br>
- param <br>
**LossMSE** <br>
