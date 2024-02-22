from flask import Flask, request, jsonify
import rembg
import importlib
import PIL
single_inference = importlib.import_module("Ladi-Vton.src.single_inference")

app = Flask(__name__)


@app.route("/init", methods=['GET'])
def init():
    global LADIVTON_SETTINGS
    global INIT_VARS
    
    LADIVTON_SETTINGS = {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-inpainting",
        "output_dir": ".\\output",  # You need to specify a valid path here
        "allow_tf32": False,  # Set to True if you want to enable TF32 on Ampere GPUs
        "seed": 1234,
        "batch_size": 1,
        "mixed_precision": None,  # Can be "no", "fp16", or "bf16"
        "enable_xformers_memory_efficient_attention": False,  # Set to True to enable
        "dresscode_dataroot": "..\\dataset\\DressCode",  # Specify if using DressCode dataset
        "vitonhd_dataroot": "..\\dataset\\vitonhd\\data",  # Specify if using VitonHD dataset
        "num_workers": 8,
        "num_vstar": 16,
        "test_order": "unpaired",  # Or "paired", you need to choose based on your requirements
        "dataset": "dresscode",  # Or "dresscode", depending on which dataset you're using
        "category": "all",  # Can be 'all', 'lower_body', 'upper_body', 'dresses'
        "use_png": False,  # Set to True if you prefer PNG over JPG
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "compute_metrics": False,  # Set to True if you want to compute metrics after generation
    }

    INIT_VARS = single_inference.initialize(**LADIVTON_SETTINGS)

@app.route("/rmbg", methods=['POST'])
def rmbg():
    try:
        id = request.form['id']
        image = request.files['image']
        output = rembg.remove(image.read())
        with open('.\\images\\output\\rmbg\\output.png', 'wb') as f:
            f.write(output)
        return 'Success'
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route("/ladivton", methods=['POST'])
def ladivton_predict():
    data = request.json
    try:
        # Pass the INIT_VARS directly to run_inference
        output = single_inference.run_inference("1", "048393_0", "048392_1", INIT_VARS=INIT_VARS, **LADIVTON_SETTINGS)
        print(f'Output: {output}')
        return output
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9090)
