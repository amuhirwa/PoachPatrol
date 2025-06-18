# Brief Description:
The problem I am trying to solve is poaching in animal habitats which are so unbelievably large that they become almost impossible to fully monitor for the guards there, making it easy for poachers to attack animals without being scared of being caught. To solve this I plan on building an alerting system that listens to audio, makes a prediction using my models and then sends an alert to wildlife rangers when human activity is heard, so they can get there in time to stop them or arrest them. 
I used a dataset of 2000 audio files of environmental sounds that could occur in those such habitats, the sounds range from Animal sounds, Natural sounds, Human sounds, Interior sounds, Exterior sounds. But I have refined the classes into 3 I need namely Human activity, Animal Sounds, and Ambient Sounds.

Dataset Link: https://github.com/karolpiczak/ESC-50

Video Presentation Link: https://youtu.be/o1y10BRVG2M

# Summary Table

| Model  | Optimizer | Regularization | Early Stopping | Dropout | Learning Rate | Num of Layers       | Accuracy | Precision | Recall | F1-Score | Best Epoch | Training Loss | Validation Loss |
|--------|-----------|----------------|----------------|---------|---------------|-------------------|----------|-----------|--------|----------|------------|---------------|------------------|
| **model_1**| Adam      | None           | False          | 0.0     | 0.001         | 10     | **0.8036** | 0.8033    | 0.8036 | 0.8032   | 30         | 0.0058       | 1.2060          |
| **model_2**| RMSprop   | L2 (0.01)      | True (pat=5)   | 0.4     | 0.0005        | 10     | 0.6667   | 0.6847    | 0.6667 | 0.6608   | 18         | 0.9430       | 0.9154          |
| **model_3**| Adam      | L2 (0.01)      | True (pat=10)  | 0.2     | 0.001         | 12     | 0.7024   | 0.7107    | 0.7024 | 0.6999   | 20         | 0.8295       | 0.8558          |
| **model_4**| SGD       | L1_L2          | True (pat=5)   | 0.5     | **0.01**      | 12     | **0.7976** | 0.8076    | 0.7976 | 0.7999   | 30         | 1.4965       | 1.8144          |
| **model_5** (RF) | n/a   | n/a            | n/a            | n/a     | n/a           | n/a               | 0.6905   | 0.7204    | 0.6905 | 0.6873   | n/a        | n/a          | n/a              |

## Summary of results
1. **Model 1**: Model one had everything set to **defaults** (learning rate of 0.001 for Adam) with no optimization or techniques to improve the model used. It achieved an accuracy of **0.8036** which is the highest but it is highly deceiving because of how **overfitted** it is. Due to having **no early stopping, dropout, or regularization**, it went past it's minima and kept memorizing data which is evident by the huge gap between the final losses with training loss being at **0.0058** and validation loss being at **1.2060**.
As for the error metrics the precision and recall are again high indicating it only shows about **80%** of false positives and false negatives. But due to how overfitted it is I wouldn't trust it with outside data as it would probably not be able to generalize.

2. **Model 2**: Model 2, I used some techniques to try to improve the model which greatly helped with the overfitting from the first model. I also used a different optimizer namely **RMSprop**. I used the **l2 regularizer** to penalize any large weights that may arise and thus prevent overfitting, I also used early stopping with a **patience of 5** to stop when the model stops progressing and not just keep memorizing it if it's not learning, I used a **relatively high dropout** to turn off some neurons through the training but keep the training efficiency by it not being too high.
This resulted in a model that did not overfit shown by the close losses (0.9154, 0.9430) but also didn't perform as well as the others with an accuracy of 0.667. Recall are also close by with a 1/3 of predictions being false negatives and a bit better with the precision showing 32% instead being false positives. I believe this might be due to having a short patience and high dropout, while also having a low learning rate so it was learning slower and might've stopped when still learning.

3. **Model 3**: Model 3, I believed performed **the best**. I used Adam again as it combines RMSprop learning rate adjustion and momentum (tracks the mean of gradients), which I believe helped. I also had a **higher learning rate** than model 2 to allow it proper improvement every epoch, and **reduced the dropout** because it might have been too harsh dropping almost half of the neurons after every layer. I also brought the early stopping **patience up to 10** to give it more time to make sure it has reached the global minima.
From the results we can see it did not overfit, the difference between it's losses is small (0.8558, 0.8295), the smallest among the models. Looking at the classification report, the ones I care about the most are the ones concerning the human activity because that it was what would need an alert, the precision on human activity is **0.72** meaning only **28%** of the time when it guesses human activity is it something else which is alot better than model 2. And looking at the recall on human activity it is **0.68** which means a good amount of the time when it is human activity it will guess it instead of the others.

4. **Model 4**: For the fourth model even though accuracy appears high there was also some strong **overfitting** towards the end. I believe this is due to how sgd works, the learning rate did not adapt in time leading to it continuing down the wrong path and the learning rate was quite high which could make it even worse when it overshoots the minima, especially **without an implemented learning rate reducer**. Looking at the confusion matrix I'd be led to believe it performed well on the test data to with few false positives 25% (meaning being correct most of the time when it guesses human activity) and relatively few false negatives 33% but this is still not good enough considering the overfitting. It’s possible that the model learned patterns unique to the audio recording conditions, such as noise or specific characteristics of the data, rather than learning more generalizable features.

5. **Model 5**: I used Random Forest because when I tried to find a decent algorithm for audio classification especially with multiple classes that came up the most because it builds multiple decision trees and combines their results to improve accuracy and robustness.
I used GridSearchCV which is a hyperparameter tuning technique in scikit-learn that helps you find the best combination of hyperparameters for a machine learning model. 
And these are the hyperparameters I got: Best Hyperparameters:
max_depth: 20
max_features: sqrt
min_samples_leaf: 2
min_samples_split: 2
n_estimators: 100
My final results where surprisingly promising with a decent accuracy at 0.6905. But for the confusion matrix we could see it most often just guessed human activity, with recall on human activity being at 0.87 meaning it rarely misses them, which was good for our use case but not necessarily the best. I also checked the training accuracy with suspicion of overfitting and it came out to 0.9936 which means it heavily overfitted so all of those results are still not good.

## Insights:
### Which combination worked better?
- Based on what we discussed I believed model 3 (Adam, learning_rate=0.001, dropout=0.2, early_stopping=True(patience=10), regularization=L2 (0.01)) performed the best having the least difference between losses which shows it didn't overfit while still having relatively high accuracy (0.7024) and good error metrics ().
### Which implementation worked better: ML Algorithm or Neural Network?
- Based on the results, the Neural Network model (Model 3) worked better overall compared to the Random Forest (Model 5). Although Model 1 achieved the highest accuracy, it was highly overfitted, which made it unreliable for generalization. Model 3 showed the best balance between accuracy, precision, and recall, with minimal overfitting, making it more robust for real-world use, particularly for detecting human activity. While the Random Forest model performed surprisingly decently, its high training accuracy and overfitting issues indicated that it may struggle to generalize to unseen data, making Model 3 the most reliable choice for practical deployment. And even without the overfitting, the Random Forest was still nowhere near the best in terms of any metrics.

##  Running the Notebook

### 1. Clone the Repository

```bash
git clone https://github.com/amuhirwa/PoachPatrol.git
cd PoachPatrol
```
### 2. Open the Notebook
```bash
  jupyter notebook notebook.ipynb
```

### 3. Run All Cells in Sequence

The notebook is modularized, with each model instance (Instance 1 to Instance 5) in its own section.

It includes audio to mel spectogram conversion, data preprocessing, model training, and evaluation.

Make sure to download the dataset and bring the audio directory to the project root directory.


### 4. Loading the best saved model

Go to the before last cell and edit the model_path and also choose the item to predict for in the make_predictions function call.
```python
model_path = "./saved_model/model_1_unoptimized.keras"
make_predictions(model_path, np.expand_dims(X_test[1], axis=0), return_labels=True)
```