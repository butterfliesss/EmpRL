import torch
from models import SeperateT5EncoderClassifier


class EmpathyClassifer():
    def __init__(self):
        super(EmpathyClassifer, self).__init__()

        self.emo_classifier_model = SeperateT5EncoderClassifier("base", 3)
        self.emo_classifier_model.load_state_dict(
            torch.load("saved/empathy/1667030885/model.pt"))

        self.exp_classifier_model = SeperateT5EncoderClassifier("base", 3)
        self.exp_classifier_model.load_state_dict(
            torch.load("saved/empathy/1667009478/model.pt"))

        self.int_classifier_model = SeperateT5EncoderClassifier("base", 3)
        self.int_classifier_model.load_state_dict(
            torch.load("saved/empathy/1667016350/model.pt"))

        self.emo_classifier_model.cuda()
        self.exp_classifier_model.cuda()
        self.int_classifier_model.cuda()

    def predict(self, context, response):
        self.emo_classifier_model.eval()
        self.exp_classifier_model.eval()
        self.int_classifier_model.eval()

        with torch.no_grad():
            logit_emo = self.emo_classifier_model(context, response)
            logit_exp = self.exp_classifier_model(context, response)
            logit_int = self.int_classifier_model(context, response)

        predict_emo = torch.argmax(logit_emo, 1).data.cpu().numpy()
        predict_exp = torch.argmax(logit_exp, 1).data.cpu().numpy()
        predict_int = torch.argmax(logit_int, 1).data.cpu().numpy()
        
        return predict_emo, predict_exp, predict_int