import torch
from models import SeperateT5EncoderClassifier

class Rewards():
    """
    compute the rewards, please note that the parameters in this class are fixed.
    """
    def __init__(self):
        super(Rewards, self).__init__()
        
        print("use original classifier model")
        self.emo_classifier_model = SeperateT5EncoderClassifier("base", 3)#emo
        self.emo_classifier_model.load_state_dict(
            torch.load("saved/empathy/1667030885/model.pt"))

        self.exp_classifier_model = SeperateT5EncoderClassifier("base", 3)#exp
        self.exp_classifier_model.load_state_dict(
            torch.load("saved/empathy/1667009478/model.pt"))

        self.int_classifier_model = SeperateT5EncoderClassifier("base", 3) #int
        self.int_classifier_model.load_state_dict(
            torch.load("saved/empathy/1667016350/model.pt"))
        
        self.emo_classifier_model.cuda()
        self.exp_classifier_model.cuda()
        self.int_classifier_model.cuda()

        self.empathy_loss_function = torch.nn.CrossEntropyLoss().cuda()


    def predict_empathy(self, context, generated_response):
        """
        return predicted empahty labels from three aspects.
        """
        self.emo_classifier_model.eval()
        self.exp_classifier_model.eval()
        self.int_classifier_model.eval()
        
        with torch.no_grad():
            logit_emo = self.emo_classifier_model(context, generated_response)
            logit_exp = self.exp_classifier_model(context, generated_response)
            logit_int = self.int_classifier_model(context, generated_response)

        return logit_emo, logit_exp, logit_int
    
    def calc_empathy_score(self, context, generated_response, emo_label, exp_label, int_label): # one context
        """
        this for test, we we design the appropriate rewarad for empathy.
        """
        emo_pred, exp_pred, int_pred = self.predict_empathy(context, generated_response)
        
        softmax = torch.nn.Softmax(dim=1)

        emo_probs = softmax(emo_pred)
        emo_log_probs = torch.log(emo_probs)
        emo_loss_per_sample = -emo_log_probs.gather(1, emo_label.view(-1, 1))
        emo_loss_per_sample = emo_loss_per_sample

        exp_probs = softmax(exp_pred)
        exp_log_probs = torch.log(exp_probs)
        exp_loss_per_sample = -exp_log_probs.gather(1, exp_label.view(-1, 1))
        exp_loss_per_sample = exp_loss_per_sample

        int_probs = softmax(int_pred)
        int_log_probs = torch.log(int_probs)
        int_loss_per_sample = -int_log_probs.gather(1, int_label.view(-1, 1))
        int_loss_per_sample = int_loss_per_sample

        emo_loss = emo_loss_per_sample
        exp_loss = exp_loss_per_sample
        int_loss = int_loss_per_sample

        empathy_loss = emo_loss + exp_loss + int_loss
        score = torch.exp(-empathy_loss)
        return score

    def forward(self, context, generated_response, 
                emo_label, exp_label, int_label):
        emp_score = self.calc_empathy_score(context, generated_response, 
                                            emo_label, exp_label, int_label)
        return emp_score
        
