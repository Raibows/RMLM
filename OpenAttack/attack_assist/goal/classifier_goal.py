from .base import AttackGoal

class ClassifierGoal(AttackGoal):
    def __init__(self, target, targeted):
        self.target = target
        self.targeted = targeted
    
    @property
    def is_targeted(self):
        return self.targeted

    def check(self, adversarial_sample, prediction):
        # prediction = -1 means been detected by the victim model
        if prediction == -1:
            return False # attack failed
        if self.targeted:
            return prediction == self.target
        else:
            return prediction != self.target