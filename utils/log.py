import os 
import sys
import logging 
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import wraps

# def log_to_file(path, log=True):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             if log:
#                 # Set up logging to both file and terminal
#                 logger = logging.getLogger(func.__name__)
#                 logger.setLevel(logging.INFO)

#                 # Create a file handler for logging to the specified file
#                 file_handler = logging.FileHandler(path)
#                 file_handler.setLevel(logging.INFO)
#                 file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#                 file_handler.setFormatter(file_formatter)

#                 # Create a stream handler for logging to the terminal
#                 stream_handler = logging.StreamHandler(sys.stdout)
#                 stream_handler.setLevel(logging.INFO)
#                 stream_formatter = logging.Formatter('%(message)s')
#                 stream_handler.setFormatter(stream_formatter)

#                 # Add both handlers to the logger
#                 logger.addHandler(file_handler)
#                 logger.addHandler(stream_handler)

#                 # Redirect stdout to a custom stream that writes to both file and terminal
#                 class DualStream:
#                     def __init__(self, file_handler, stream_handler):
#                         self.file_handler = file_handler
#                         self.stream_handler = stream_handler

#                     def write(self, message):
#                         # Write to file
#                         self.file_handler.emit(logging.LogRecord(
#                             name=func.__name__,
#                             level=logging.INFO,
#                             pathname=__file__,
#                             lineno=0,
#                             msg=message.strip(),
#                             args=None,
#                             exc_info=None
#                         ))
#                         # Write to terminal
#                         self.stream_handler.stream.write(message)

#                     def flush(self):
#                         self.stream_handler.stream.flush()

#                 # Replace sys.stdout with the custom DualStream
#                 original_stdout = sys.stdout
#                 sys.stdout = DualStream(file_handler, stream_handler)

#                 try:
#                     # Call the original function
#                     result = func(*args, **kwargs)
#                     logger.info(f"Function '{func.__name__}' executed successfully.")
#                     return result
#                 except Exception as e:
#                     logger.error(f"Function '{func.__name__}' raised an exception: {e}")
#                     raise
#                 finally:
#                     # Restore the original stdout
#                     sys.stdout = original_stdout
#                     # Remove handlers to avoid duplicate logs in future calls
#                     logger.removeHandler(file_handler)
#                     logger.removeHandler(stream_handler)
#             else:
#                 # If log is False, just call the function normally
#                 return func(*args, **kwargs)

#         return wrapper
#     return decorator


def log_to_file(path, log=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if log:
                # Set up logging to both file and terminal
                logger = logging.getLogger(func.__name__)
                logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

                # Create a file handler for logging to the specified file
                file_handler = logging.FileHandler(path)
                file_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
                file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)

                # Create a stream handler for logging to the terminal
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
                stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
                stream_handler.setFormatter(stream_formatter)

                # Add both handlers to the logger
                logger.addHandler(file_handler)
                logger.addHandler(stream_handler)

                # Redirect stdout to a custom stream that writes to both file and terminal
                class DualStream:
                    def __init__(self, file_handler, stream_handler):
                        self.file_handler = file_handler
                        self.stream_handler = stream_handler

                    def write(self, message):
                        if message.strip():  # Avoid logging empty messages
                            # Determine the log level based on the message content
                            if "ERROR" in message:
                                level = logging.ERROR
                            elif "WARNING" in message:
                                level = logging.WARNING
                            else:
                                level = logging.INFO

                            # Write to file
                            self.file_handler.emit(logging.LogRecord(
                                name=func.__name__,
                                level=level,
                                pathname=__file__,
                                lineno=0,
                                msg=message.strip(),
                                args=None,
                                exc_info=None
                            ))

                        # Always write to terminal
                        self.stream_handler.stream.write(message)

                    def flush(self):
                        self.stream_handler.stream.flush()

                # Replace sys.stdout with the custom DualStream
                original_stdout = sys.stdout
                sys.stdout = DualStream(file_handler, stream_handler)

                try:
                    # Call the original function
                    result = func(*args, **kwargs)
                    logger.info(f"Function '{func.__name__}' executed successfully.")
                    return result
                except Exception as e:
                    logger.error(f"Function '{func.__name__}' raised an exception: {e}")
                    raise
                finally:
                    # Restore the original stdout
                    sys.stdout = original_stdout
                    # Remove handlers to avoid duplicate logs in future calls
                    logger.removeHandler(file_handler)
                    logger.removeHandler(stream_handler)
            else:
                # If log is False, just call the function normally
                return func(*args, **kwargs)

        return wrapper
    return decorator
        
def plot_train_stats(train_loss_1, train_acc_1, train_grad_norms_1, train_loss_2, train_acc_2, train_grad_norms_2, directory, acc_low=0):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,2.5), sharey='row')
    axes[0].plot(np.array(train_loss_1), label="SGD")
    axes[0].plot(np.array(train_loss_2), label="SVRG")
    axes[0].set_title("Train loss")
    axes[0].legend()
    axes[1].plot(np.array(train_acc_1), label="SGD")
    axes[1].plot(np.array(train_acc_2), label="SVRG")
    axes[1].set_ylim(acc_low, 1)
    axes[1].set_title("Train Accuracy")
    axes[1].legend()
    axes[2].plot(np.array(train_grad_norms_1), label="SGD")
    axes[2].plot(np.array(train_grad_norms_2), label="SVRG")
    axes[2].set_title("Train Gradient Norms")
    # set a log yticks
    axes[2].set_yscale('log')
    axes[2].legend()
    # add a global x axis
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'train_stats.pdf'))
    plt.close()
    
class AverageCalculator():
    def __init__(self):
        self.reset() 
    
    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / float(self.count)
        
def setup_output_directory(args):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"{timestamp}_{args.optimizer}_{args.dataset}_{args.nn_model}_Temperature{str(args.temperature)}_lr{str(args.lr)}"
    
    if args.log:
        if args.exp_name != "":
            model_name = f"{args.exp_name}_{model_name}"
        log_dir = os.path.join(args.output_dir, model_name)
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)
    else:
        # log_dir = current directory
        log_dir = os.path.join(os.getcwd(), model_name)
    return log_dir



