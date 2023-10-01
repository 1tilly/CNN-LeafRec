#!/usr/bin python

import sys
import pygtk
import gtk
import matplotlib.pyplot as plt

from LeafDataset import load_and_split_dataset
from train import train_model, eval_model
from ResNet import ResNet
from CNN_noReg import CNN

class Runner():

    def main(self):
        gtk.main()


    def __init__(self):
        self.window = gtk.Window()
        self.window.connect("destroy", lambda wid: gtk.main_quit())
        self.window.connect("delete_event", lambda a1, a2: gtk.main_quit())
        self.window.set_border_width(10)
        self.window.set_title("Leaf Recognition")


        button = gtk.Button("first button")
        button.connect("clicked", self.start, "first Button")
        button.set_size_request(200,30)
        quitbutton = gtk.Button("Quit", gtk.STOCK_CLOSE, 'Close Button')
        quitbutton.connect("clicked", gtk.main_quit)
        quitbutton.set_size_request(200,30)

        boxx = gtk.HBox()
        boxx.pack_start(button, fill=False)
        boxx.pack_start(quitbutton, fill=False)
        boxy = gtk.VBox()
        boxy.pack_start(boxx, fill=False)
        self.window.add(boxy)
        self.window.show()
        boxx.show()
        boxy.show()
        quitbutton.show()
        button.show()

    def start(self,widget, data=None):
        print("Training and Evaluating Models...")

        # Train and evaluate CNN
        self.train_evaluate_plot(CNN, "CNN")

        # Train and evaluate ResNet
        self.train_evaluate_plot(ResNet, "ResNet")

    def load_data(self):
        # Load the data
        self.train_set, self.test_set, self.eval_set, self.num_classes = load_and_split_dataset()

    def train_evaluate_plot(self, model_cls, model_name):
        # Train the model
        model = self.train(model_cls)

        # Evaluate the model
        accuracy, roc_auc = self.eval(model)

        # Plot the results
        self.plot_results(accuracy, roc_auc, model_name)

    def train(self, model_cls):


        # Initialize the model
        model = model_cls(num_classes=self.num_classes)

        # Compile the model and get the training function
        train_fn = model.compile()  

        # Train the model
        train_set_x, train_set_y = self.train_set
        train_model(train_fn, train_set_x, train_set_y)

    def eval(self, model):
        eval_fn = model.compile_eval()
        roc_fn = model.compile_roc()

        # Evaluate the model (assuming you have eval_set_x and eval_set_y)
        eval_set_x, eval_set_y = self.eval_set
        accuracy, roc_auc = eval_model(eval_fn, roc_fn, eval_set_x, eval_set_y)

        return accuracy, roc_auc

    def plot_results(self, accuracy, roc_auc, model_name):
        plt.figure()
        plt.title(f"{model_name} Evaluation")
        plt.bar(["Accuracy", "ROC AUC"], [accuracy, roc_auc])
        plt.show()

if __name__ == "__main__":
    run = Runner()
    run.main()