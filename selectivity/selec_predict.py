import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot

def doML (data, output, ts, rs):
    print ("Output for " + output)
    x_train, x_test, y_train, y_test = train_test_split(data.drop(output,axis=1), 
           data[output], test_size=ts, 
            random_state=rs)

    model = GaussianNB()
    model.fit(x_train, y_train)
    
    #AUC calculation 
    probs = model.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
       
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.show()

    #Confusion matrix
    predictions = model.predict(x_test)
    
    f = open("confusion.txt", "w")
    f.write(classification_report(y_test,predictions))
    conm = confusion_matrix(y_test, predictions)
    f.write("\nTrue positive:" + str(conm[0][0]))
    f.write("\nTrue negative:" + str(conm[1][1]))
    f.write("\nFalse positive:" + str(conm[0][1]))
    f.write("\nFalse negative:" + str(conm[1][0]))

    #F1 score
    Precision = conm[0][0]/float(conm[0][0]+conm[0][1])
    Recall = conm[0][0]/float(conm[0][0]+conm[1][0])
    
    F1_score = 2*(Recall*Precision)/(Recall+Precision)
    
    f.write("\nF1 score:" + str(F1_score))
    
    f.write("\nAUC:" + str(auc))

    f.close()
    


data = pd.read_csv("data.csv")

data = data.drop('ch4', axis=1)
data = data.drop('h2', axis=1)

doML(data, 'selec', 0.5, 2)
