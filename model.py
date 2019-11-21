
from argparse import ArgumentParser
import logging
import json
import math
import copy

import matplotlib.pyplot as pyplot

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program uses a roofline performance "
        "model to predict the training time for a speech recognition system.")

    parser.add_argument("-c", "--config-path", default="",
        help = "The input path to the config file describing "
               "the problem and system.")
    parser.add_argument("-p", "--plot-roofline", default=False, action="store_true",
        help = "Plot the roofline.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed "
               "messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    run_model(arguments)

def run_model(arguments):
    config = load_config(arguments)

    application_model = get_application_model(config)
    machine_model = get_machine_model(config)

    if arguments["plot_roofline"]:
        pyplot.figure()
        plot_roofline(application_model, machine_model)
        plot_sweep(config)

        pyplot.xlabel("Reuse (flops/byte)")
        pyplot.ylabel("TFLOP/s")
        pyplot.legend()
        pyplot.show()

    time = run_roofline(application_model, machine_model, config["system"]["processor-count"])

    logger.info("Runtime is", time / 3600.0, "hours")

def run_roofline(application_model, machine_model, processor_count):
    bandwidth_time = application_model["total-bytes"] / machine_model["bytes-per-second"]
    # Assume we use NCCL all-reduce, we estimate all-reduce time of n processors based on
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    bandwidth_time *= 2.0 * (processor_count - 1) / processor_count

    compute_time   = application_model["total-flops"] / machine_model["flops-per-second"]
    ff_compute_time = compute_time * 0.4 # empirical ratio of how much time does ff take
    bw_compute_time = compute_time - ff_compute_time

    result = ff_compute_time + max(bw_compute_time, bandwidth_time) # comm overlap only with comp

    return result

def plot_roofline(application_model, machine_model):
    reuse = application_model["total-flops"] / application_model["total-bytes"]

    reuse_range = [2.0 * reuse * i / 100.0 for i in range(100)]
    flops_per_second = [ min(machine_model["bytes-per-second"] * reuse,
                             machine_model["flops-per-second"]) / 1e12
                         for reuse in reuse_range ]

    pyplot.plot(reuse_range, flops_per_second)

def plot_sweep(config):
    for sweep in config["sweep"]:
        key = sweep[0]
        values = sweep[1]

        new_config = copy.deepcopy(config)

        for value in values:
            new_config[key[0]][key[1]] = value

            plot_point(new_config, key[1] + "-" + str(value))

def plot_point(config, name):
    application_model = get_application_model(config)
    machine_model = get_machine_model(config)

    reuse = application_model["total-flops"] / application_model["total-bytes"]

    tflops_per_second = min(machine_model["bytes-per-second"] * reuse,
                            machine_model["flops-per-second"]) / 1e12

    pyplot.plot(reuse, tflops_per_second, 'o', label=name)

def get_application_model(config):
    application_model = {}

    application_model["total-bytes"] = compute_total_bytes(config)
    application_model["total-flops"] = compute_total_flops(config)

    return application_model

def get_machine_model(config):
    machine_model = {}

    machine_model["bytes-per-second"] = config["system"]["processor-count"] * config["system"]["bytes-per-second"]
    machine_model["flops-per-second"] = config["system"]["processor-count"] * config["system"]["flops-per-second"]

    return machine_model

def compute_total_bytes(config):
    frame_size = config["application"]["frame-size"]

    frames = get_frame_count(config)

    # 16-bit samples
    input_bytes_per_frame = frame_size * 2

    # 32-bit weights
    model_bytes_per_frame = get_parameter_count(config) * 4 // (config["model"]["reuse"] * config["application"]["batch-size"])

    # 16-bit activations
    activation_bytes_per_frame = config["model"]["layer-count"] * get_activations_per_layer(config) * 2

    bytes_per_frame = model_bytes_per_frame + input_bytes_per_frame

    return frames * bytes_per_frame

def compute_total_flops(config):
    frames = get_frame_count(config)

    flops_per_frame = 2 * 3 * get_parameter_count(config)

    return frames * flops_per_frame

def get_parameter_count(config):
    return config["model"]["parameter-count"]

def get_activations_per_layer(config):
    return math.sqrt(get_parameter_count(config) / config["model"]["layer-count"])

def get_frame_count(config):
    epochs = config["application"]["epochs"]
    hours = config["application"]["hours-of-speech"]
    sample_rate = config["application"]["sample-rate"]
    frame_step = config["application"]["frame-step"]
    batch_size = config["application"]["batch-size"]

    samples = hours * sample_rate * 3600

    frames_per_epoch = (samples + frame_step - 1) // frame_step

    frames = frames_per_epoch * epochs

    return frames

def load_config(arguments):
    return json.load(open(arguments["config_path"]))

def setup_logger(arguments):

   if arguments["verbose"]:
       logger.setLevel(logging.DEBUG)
   else:
       logger.setLevel(logging.INFO)

   ch = logging.StreamHandler()
   ch.setLevel(logging.DEBUG)

   # create formatter
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

   # add formatter to ch
   ch.setFormatter(formatter)

   # add ch to logger
   logger.addHandler(ch)

main()
