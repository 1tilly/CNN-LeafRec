#!/usr/bin python

import sys
import pygtk
import gtk
import ConvolutionalNetwork


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
        print "started"
        ConvolutionalNetwork.



run = Runner()
run.main()