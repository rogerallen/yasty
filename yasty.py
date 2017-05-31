#!/usr/bin/env python3
"""
yasty.py - yet another stylistic transfer deep learning image creator.

by Roger Allen
Heavily based on code from Lesson 8 of Fast.ai Practical Deep Learning

Written to work with this anaconda environment.
https://gist.github.com/rogerallen/f96d0dce129d18ee9351737b20b1f521/c244b37b3e59fad0ae3eafbb32269855f7f9e2f7
Anaconda Python 3.5, Keras 1.2.2, Tensorflow 1.0.1

activate dl2_p35k12tf10

Apache License, Version 2.0, January 2004, http://www.apache.org/licenses/

"""
import os
import sys
import logging
import argparse
import configparser
import json

# remove tensorflow spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
from PIL import Image
from vgg16_avg import VGG16_Avg
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

# ======================================================================
# FIXME - grok this
def gram_matrix(x):
    # We want each row to be a channel, and the columns to be flattened x,y locations
    features = keras.backend.batch_flatten(keras.backend.permute_dimensions(x, (2, 0, 1)))
    # The dot product of this with its transpose shows the correlation
    # between each pair of channels
    return keras.backend.dot(features, keras.backend.transpose(features)) / x.get_shape().num_elements()

def style_loss(x, targ):
    return keras.metrics.mse(gram_matrix(x), gram_matrix(targ))

rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
deproc = lambda x,s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)
# FIXME do blotchy random
rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape)/100

# FIXME - grok this
def solve_image(eval_obj, output_path, niter, x, shp):
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127,127)
        filename = output_path+'_%02d.png'%(i)
        logging.info('%s loss: %.2f'%(filename, min_val))
        imsave(filename, deproc(x.copy(), shp)[0])
    return x

# FIXME - grok this
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
        style_layers = [outputs['block{}{}'.format(o,self.args.style_suffix)]
                        for o in range(self.args.style_block_min,self.args.style_block_max)]
        content_name = self.args.content_layer
        content_layer = outputs[content_name]
        style_model = keras.models.Model(model.input, style_layers)
        style_targs = [keras.backend.variable(o) for o in style_model.predict(self.style_image_array)]
        content_model = keras.models.Model(model.input, content_layer)
        content_targ = keras.backend.variable(content_model.predict(self.input_image_array))
        style_wgts = self.args.style_weights
        loss = sum(style_loss(l1[0], l2[0])*w
                   for l1,l2,w in zip(style_layers, style_targs, style_wgts))
        loss += keras.metrics.mse(content_layer, content_targ)/10 # ??? FIXME: why div 10
        grads = keras.backend.gradients(loss, model.input)
        self.transfer_fn = keras.backend.function([model.input], [loss]+grads)

    def evaluate(self):
        evaluator = Evaluator(self.transfer_fn, self.input_image_array.shape)
        x = rand_img(self.input_image_array.shape)
        x = solve_image(evaluator, self.args.output_image, self.args.num_iterations, x, self.input_image_array.shape)

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
            default=self.config.get("options", "verbose", 0),
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
            default=self.config.get("options", "output_image", None),
            help="path to output image (REQUIRED).  Will add iteration number.png to this."
        )
        parser.add_argument(
            "-s","--style",
            dest="style_image",
            default=None,
            help="path to style image (REQUIRED)"
        )
        parser.add_argument(
            "-n","--num_iterations",
            dest="num_iterations",
            default=self.config.get("options", "num_iterations", 9),
            help="Number of iterations to do.  Will save an image for each one."
        )
        parser.add_argument(
            "--style_block_min",
            dest="style_block_min",
            default=self.config.get("layers", "style_block_min", 1),
            help="min layer for style block%%d."
        )
        parser.add_argument(
            "--style_block_max",
            dest="style_block_max",
            default=self.config.get("layers", "style_block_max", 6),
            help="max layer for style block%%d."
        )
        parser.add_argument(
            "--style_suffix",
            dest="style_suffix",
            default=self.config.get("layers", "style_suffix", "_conv1"),
            help="suffix for style blocks."
        )
        parser.add_argument(
            "--style_weights",
            dest="style_weights",
            default=self.config.get("layers", "style_weights", "[0.1,0.2,0.2,0.2,0.2,0.3]"),
            help="suffix for style blocks."
        )
        parser.add_argument(
            "--content_layer",
            dest="content_layer",
            default=self.config.get("layers", "content_layer", "block4_conv1"),
            help="layer for content error check."
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
        # convert weights string to an array
        print(self.args.style_weights)
        self.args.style_weights   = json.loads(self.args.style_weights)
        self.args.style_block_min = int(self.args.style_block_min)
        self.args.style_block_max = int(self.args.style_block_max)
        self.args.num_iterations  = int(self.args.num_iterations)

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
