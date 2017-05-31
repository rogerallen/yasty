#!/usr/bin/env python3
"""
yasty.py - yet another stylistic transfer deep learning thing

Heavily based on code from Lesson 8 of Fast.ai Practical Deep Learning

Written to work with this anaconda environment.
https://gist.github.com/rogerallen/f96d0dce129d18ee9351737b20b1f521/c244b37b3e59fad0ae3eafbb32269855f7f9e2f7
Anaconda Python3.5 Keras1.2.2 Tensorflow1.0.1

activate dl2_p35k12tf10

"""
import os
import sys
import logging
import argparse
import configparser
#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import keras.backend as K
from keras.models import Model
import numpy as np
from PIL import Image
from vgg16_avg import VGG16_Avg
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

# ======================================================================
def gram_matrix(x):
    # We want each row to be a channel, and the columns to be flattened x,y locations
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # The dot product of this with its transpose shows the correlation
    # between each pair of channels
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()

def style_loss(x, targ):
    return keras.metrics.mse(gram_matrix(x), gram_matrix(targ))

rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
deproc = lambda x,s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)
# FIXME do blotchy random
rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape)/100

def solve_image(eval_obj, niter, x, shp):
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127,127)
        print('Current loss value:', min_val)
        # FIXME
        imsave('iteration_%d.png'%(i), deproc(x.copy(), shp)[0])
    return x

class Evaluator(object):
    def __init__(self, f, shp):
        self.f, self.shp = f, shp

    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x):
        return self.grad_values.flatten().astype(np.float64)

# ======================================================================
class Config(object):
    """Config Class.  Use a configuration file to control your program
    when you have too much state to pass on the command line.
    Reads the <program_name>.cfg or ~/.<program_name>.cfg file for
    configuration options.
    Handles booleans, integers and strings inside your cfg file.
    """
    def __init__(self,program_name=None):
        self.config_parser = configparser.ConfigParser()
        if not program_name:
            program_name = os.path.basename(sys.argv[0].replace('.py',''))
            self.config_parser.read([program_name+'.cfg',
                                   os.path.expanduser('~/.'+program_name+'.cfg')])
    def get(self,section,name,default):
        """returns the value from the config file, tries to find the
        'name' in the proper 'section', and coerces it into the default
        type, but if not found, return the passed 'default' value.
        """
        try:
            if type(default) == type(bool()):
                return self.config_parser.getboolean(section,name)
            elif type(default) == type(int()):
                return self.config_parser.getint(section,name)
            else:
                return self.config_parser.get(section,name)
        except:
            return default

# ======================================================================
class Application(object):
    def __init__(self,argv):
        self.config = Config()
        self.parse_args(argv)
        self.adjust_logging_level()

    def setup_images(self,input_filename, style_filename):
        # read w,h,3(rgb) array
        self.input_image = Image.open(input_filename)
        self.style_image = Image.open(style_filename)
        # resize style to be a big as input, then crop
        ratio = max(self.input_image.width/self.style_image.width, self.input_image.height/self.style_image.height)
        logging.info("resizing style image by %.2f"%(ratio))
        self.style_image = self.style_image.resize((int(self.style_image.width*ratio),int(self.style_image.height*ratio)),Image.BICUBIC)
        # now crop to match
        logging.debug("1 input w,h = %d,%d"%(self.input_image.width,self.input_image.height))
        logging.debug("  style w,h = %d,%d"%(self.style_image.width,self.style_image.height))
        if self.style_image.width > self.input_image.width:
            delta = self.style_image.width - self.input_image.width
            a = int(delta/2)
            b = delta - a
            logging.debug("crop %d %d from horizontal"%(a,b))
            self.style_image = self.style_image.crop((a,0,self.style_image.width - b,self.style_image.height))
            self.style_image.load() # ?
        elif self.style_image.height > self.input_image.height:
            delta = self.style_image.height - self.input_image.height
            a = int(delta/2)
            b = delta - a
            logging.debug("crop %d %d from vertical"%(a,b))
            self.style_image = self.style_image.crop((0,a,self.style_image.width,self.style_image.height - b))
            self.style_image.load() # ?
        logging.debug("2 input w,h = %d,%d"%(self.input_image.width,self.input_image.height))
        logging.debug("  style w,h = %d,%d"%(self.style_image.width,self.style_image.height))
        assert(self.style_image.width == self.input_image.width)
        assert(self.style_image.height == self.input_image.height)
        # adjust to 1,w,h,3(rgb) array & remove mean
        self.input_image_array = preproc(np.expand_dims(np.array(self.input_image), 0))
        self.style_image_array = preproc(np.expand_dims(np.array(self.style_image), 0))

    def setup_models(self):
        shape = self.input_image_array.shape # both images now have same shape
        model = VGG16_Avg(include_top=False, input_shape=shape[1:])
        outputs = {l.name: l.output for l in model.layers}
        # FIXME add these as parameters
        style_layers = [outputs['block{}_conv1'.format(o)] for o in range(1,6)]
        content_name = 'block4_conv1'
        content_layer = outputs[content_name]
        style_model = Model(model.input, style_layers)
        style_targs = [K.variable(o) for o in style_model.predict(self.style_image_array)]
        content_model = Model(model.input, content_layer)
        content_targ = K.variable(content_model.predict(self.input_image_array))
        # FIXME add parameters
        style_wgts = [0.05,0.2,0.2,0.25,0.3]
        loss = sum(style_loss(l1[0], l2[0])*w
                   for l1,l2,w in zip(style_layers, style_targs, style_wgts))
        loss += keras.metrics.mse(content_layer, content_targ)/10 # ??? FIXME: why div 10
        grads = K.gradients(loss, model.input)
        self.transfer_fn = K.function([model.input], [loss]+grads)

    def evaluate(self):
        evaluator = Evaluator(self.transfer_fn, self.input_image_array.shape)
        iterations=10 # FIXME
        x = rand_img(self.input_image_array.shape)
        x = solve_image(evaluator, iterations, x, self.input_image_array.shape)

    def run(self):
        """The Application main run routine
        """
        # -v to see info messages
        logging.info("Args: {}".format(self.args))
        # -v -v to see debug messages
        #logging.debug("Debug Message")
        # we'll always see these
        #logging.warn("Warning Message")
        #logging.error("Error Message")
        self.setup_images(self.args.input_image, self.args.style_image)
        self.setup_models()
        self.evaluate()
        return 0

    def parse_args(self,argv):
        """parse commandline arguments, use config files to override
        default values. Initializes:
        self.args: a dictionary of your commandline options,
        """
        parser = argparse.ArgumentParser(description="A python3 skeleton.")
        parser.add_argument(
            "-v","--verbose",
            dest="verbose",
            action='count',
            default=self.config.get("options","verbose",0),
            help="Increase verbosity (add once for INFO, twice for DEBUG)"
        )
        parser.add_argument(
            "-i","--image",
            dest="input_image",
            default=None,
            help="path to input image (REQUIRED)"
        )
        parser.add_argument(
            "-o","--output",
            dest="output_image",
            default=None,
            help="path to input image (REQUIRED)"
        )
        parser.add_argument(
            "-s","--style",
            dest="style_image",
            default=None,
            help="path to style image (REQUIRED)"
        )
        self.args = parser.parse_args(argv)
        if self.args.input_image == None:
            print("Error: must provide input image",file=sys.stderr)
            sys.exit(1)
        if self.args.style_image == None:
            print("Error: must provide style image",file=sys.stderr)
            sys.exit(1)
        if self.args.output_image == None:
            print("Error: must provide output image",file=sys.stderr)
            sys.exit(1)

    def adjust_logging_level(self):
        """adjust logging level based on verbosity option
        """
        log_level = logging.WARNING # default
        if self.args.verbose == 1:
            log_level = logging.INFO
        elif self.args.verbose >= 2:
            log_level = logging.DEBUG
        logging.basicConfig(level=log_level)

# ======================================================================
def main(argv):
    """ The main routine creates and runs the Application.
    argv: list of commandline arguments without the program name
    returns application run status
    """
    app = Application(argv)
    return app.run()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
