import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class ModelEvaluation:
    def __init__(self):
        self.results = {}

    def evaluate_roc(self, model, x_test, y_test, model_name):
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        self.results[model_name] = (fpr, tpr, roc_auc)
        
        return roc_auc

    def plot_roc_curve(self):
        plt.figure()
        for model_name, (fpr, tpr, roc_auc) in self.results.items():
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
