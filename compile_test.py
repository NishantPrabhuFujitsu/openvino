import numpy as np
import openvino as ov
import openvino.properties.hint as hints
import openvino.properties as props
import time
import sys
import os

from transformers import AutoTokenizer
from tqdm import tqdm


def main(model_dir):
    core = ov.Core()
    core.set_property("CPU", {hints.execution_mode: hints.ExecutionMode.PERFORMANCE})
    
    config = {
        props.inference_num_threads: 64,
        hints.inference_precision: "f16", 
    }
    
    start = time.time()
    model = core.compile_model(model_dir + "/openvino_model.xml", "CPU", config)
    end = time.time()

    model_name = os.path.basename(model_dir)
    with open(f"{model_name}.txt", "a") as f:
        f.writelines(f"{end - start},")
        
    print(f"{model_name}: {end - start} sec")
    
    
if __name__ == "__main__":
    main(sys.argv[1])