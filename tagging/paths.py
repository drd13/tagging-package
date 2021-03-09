import configparser
import os

basepath = os.path.abspath(os.path.dirname(__file__))

config = configparser.ConfigParser()
config.read(os.path.join(basepath,"paths.ini"))


path_dataset = config["Paths"]["dataset"]
path_val_dataset = config["Paths"]["val_dataset"]



