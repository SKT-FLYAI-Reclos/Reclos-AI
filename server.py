from flask import Flask, request, Response
import rembg
import importlib
import PIL

cluster = importlib.import_module("clustering.clustering")
clothseg = importlib.import_module("huggingface-cloth-segmentation.single_process_backend")
ladivton = importlib.import_module("Ladi-Vton.src.single_inference")

app = Flask(__name__)

@app.route("/init", methods=['GET'])
def init():
    try:
        print("--- INITIALIZING ---")
        global LADIVTON_SETTINGS
        global CLOTHSEG_SETTINGS
        global CLUSTER_INIT_VARS
        global CLOTHSEG_INIT_VARS
        global LADIVTON_INIT_VARS
        
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
        
        CLOTHSEG_SETTINGS = {
            "mixed_precision" : None,
            "checkpoint_path": ".\\huggingface-cloth-segmentation\\model\\cloth_segm.pth",
            "input_image": ".\\images\\input\\cloth_mask\\00001_00.jpg", #".\\images\\input\\cloth-mask\\00001_00.jpg", # cloth 사진
            "output_dir": ".\\images\\output\\cloth_mask\\output.jpg" # ".\\images\\output\\cloth-mask",  # mask-image 저장 경로
        }
        
        CLUSTER_SETTINGS = {
            "dataset_path" : "..\\dataset\\DressCode",
            "upper_model_path" : ".\\clustering\\model\\features_grey_upper.npy",
            "lower_model_path" : ".\\clustering\\model\\features_grey_lower.npy",
            "cluster_upper_result_path" : ".\\clustering\\cluster_result_upper.json",
            "cluster_lower_result_path" : ".\\clustering\\cluster_result_lower.json",
        }
        
        CLUSTER_INIT_VARS = cluster.initialize(**CLUSTER_SETTINGS)
        CLOTHSEG_INIT_VARS = clothseg.initialize(**CLOTHSEG_SETTINGS)
        LADIVTON_INIT_VARS = ladivton.initialize(**LADIVTON_SETTINGS)
        
        print("--- INIT END ---")
        return Response("{'status': 'success'}", status=200, mimetype='application/json')
    
    except Exception as e:
        return Response(f"{'status': 'error : {str(e)}'}", status=500, mimetype='application/json')
    

@app.route("/rmbg", methods=['POST'])
def rmbg():
    try:
        id = request.form['id']
        image = request.files['image']
        output = rembg.remove(image.read())
        with open('.\\images\\output\\rmbg\\output.png', 'wb') as f:
            f.write(output)
        return Response("{'status': 'success'}", status=200, mimetype='application/json')
    
    except Exception as e:
        return Response(f"{'status': 'error : {str(e)}'}", status=500, mimetype='application/json')


@app.route("/ladivton", methods=['POST'])
def ladivton_predict():
    data = request.json
    try:
        # Pass the INIT_VARS directly to run_inference
        output = ladivton.run_inference("1", "048393_0", "048392_1", INIT_VARS=LADIVTON_INIT_VARS, **LADIVTON_SETTINGS)
        print(f'Output: {output}')
        return output
    
    except Exception as e:
        return Response(f"{'status': 'error : {str(e)}'}", status=500, mimetype='application/json')


@app.route("/cluster", methods=['POST'])
def clustering():
    data = request.json
    try:
        output = cluster.run_inference(data, CLUSTER_INIT_VARS)
        print(f'Output: {output}')
        return output
    
    except Exception as e:
        return Response(f"{'status': 'error : {str(e)}'}", status=500, mimetype='application/json')


@app.route("/clothseg", methods=['POST'])
def clothseg_predict():
    data = request.json
    try:
        output = clothseg.run_inference(data, CLOTHSEG_INIT_VARS)
        print(f'Output: {output}')
        return output
    
    except Exception as e:
        return Response(f"{'status': 'error : {str(e)}'}", status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9090)
