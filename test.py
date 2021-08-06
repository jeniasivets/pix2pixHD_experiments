import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np

def shift(img, shift_size, direction='horizontal'):
    '''Format of img B x C x H x W'''
    if shift_size == 0:
        return img
    if direction == 'horizontal' and shift_size < 0:
        return torch.cat([img[..., -shift_size:], img[..., shift_size:].flip(-1)], dim=-1)
    if direction == 'horizontal' and shift_size > 0:
        return torch.cat([img[..., :shift_size].flip(-1), img[..., :-shift_size]], dim=-1) 
    if direction == 'vertical' and shift_size < 0:
        return torch.cat([img[..., -shift_size:, :], img[..., shift_size:, :].flip(2)], dim=2) 
    if direction == 'vertical' and shift_size > 0:
        return torch.cat([img[..., :shift_size, :].flip(2), img[..., :-shift_size, :]], dim=2)

    
def complex_shift(img, x, y):
    return shift(shift(img, x, 'horizontal'), y, 'vertical')
    

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

text_file = open('logs.txt', 'w')
metrics = []

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:
        generated = model.inference(data['label'], data['inst'], data['image'])
        x = y = 0
        while x == 0 and y == 0:
            x, y = np.random.randint(-1, 2, size=2)
        shifted_label = complex_shift(data['label'], x, y)
        shifted_inst = complex_shift(data['inst'], x, y)
        shifted_generated = complex_shift(generated, x, y)

        alt_generated = model.inference(shifted_label, shifted_inst, data['image'])
        difference = alt_generated - shifted_generated

        valid_difference = difference

        if x < 0:
            valid_difference = valid_difference[..., :x]
        if x > 0:
            valid_difference = valid_difference[..., x:]
        if y < 0:
            valid_difference = valid_difference[..., :y, :]
        if y > 0:
            valid_difference = valid_difference[..., y:, :]
        value = 10 * torch.log10(4 / (valid_difference**2).mean())
        text_file.write(str(data['path']) + ' ' + str(value.item()) + '\n')
        metrics.append(value.item())
        
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

text_file.write('\n')
mean_result = np.mean(metrics)
text_file.write("Mean EQ-T " + str(mean_result) + '\n')

webpage.save()
text_file.close()
