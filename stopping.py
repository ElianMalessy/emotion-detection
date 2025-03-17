class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_val_acc = None
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.stop_training = False

    def __call__(self, val_acc, epoch):
        if self.best_val_acc is None:
            self.best_val_acc = val_acc
        elif val_acc < self.best_val_acc + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.stop_training = True
        else:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.epochs_no_improve = 0
