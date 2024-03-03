from flask import Flask, request, jsonify
import rembg
import importlib
import PIL
import os
import io
import base64

cluster = importlib.import_module("clustering.clustering")
clothseg = importlib.import_module("huggingface-cloth-segmentation.single_process_backend")
ladivton = importlib.import_module("Ladi-Vton.src.single_inference_backup")

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return "Hello, World!"

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
            "allow_tf32": False,  # Set to True if you want to enable TF32 on Ampere GPUs
            "seed": 1234,
            "batch_size": 1,
            "mixed_precision": None,  # Can be "no", "fp16", or "bf16"
            "enable_xformers_memory_efficient_attention": False,  # Set to True to enable
            "dresscode_dataroot": "..\\dataset\\DressCode",  # Specify if using DressCode dataset
            "vitonhd_dataroot": "..\\dataset\\vitonhd\\data",  # Specify if using VitonHD dataset
            "num_workers": 8,
            "num_vstar": 16,    #16
            "test_order": "unpaired",  # Or "paired", you need to choose based on your requirements
            "dataset": "dresscode",  # Or "dresscode", depending on which dataset you're using
            "use_png": False,  # Set to True if you prefer PNG over JPG
            "num_inference_steps": 50,  #50
            "category" : "all",
            "guidance_scale": 7.5,  #7.5
            "compute_metrics": False,  # Set to True if you want to compute metrics after generation
        }
        
        CLOTHSEG_SETTINGS = {
            "mixed_precision" : None,
            "checkpoint_path": ".\\huggingface-cloth-segmentation\\model\\cloth_segm.pth",
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
        
        print("\n--- INITIALIZATION COMPLETE ---")
        printLogo()
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        return jsonify({'status': f'error : {str(e)}'}), 500
    

@app.route("/rmbg", methods=['POST'])
def rmbg():
    # try:
    id = request.form['id']
    image = request.files['image']
    print(f'rmbg {id}')

    dir_path = f'.\\images\\runtime\\{id}'
    output_path = f'{dir_path}\\rmbg'
    
    if os.path.exists(dir_path):
        return jsonify({'status': 'error : file already exists'})
    else:
        os.makedirs(dir_path)
    
    output_image = rembg.remove(image.read())
    with open(f'{output_path}.png', 'wb') as f:
        f.write(output_image)
        
    im = PIL.Image.open(io.BytesIO(output_image))
    bg = PIL.Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, im)
    bg.save(f'{output_path}.jpg', 'JPEG')

    return jsonify({'status': 'success', 'path': f'{output_path}.png', 'image': base64.b64encode(output_image).decode('utf-8')})

    """
    except Exception as e:
        return jsonify({'status': f'error : {str(e)}'}), 500"""


@app.route("/ladivton", methods=['POST'])
def ladivton_predict():
    data = request.json
    print(f'ladivton post body: {data}')
    
    id = data['id']     # 1
    reference_id = data['reference_id'] # 048393 (model id in dataset)
    index = data['index']
    
    save_dir = f'.\\images\\runtime\\{id}'
    cloth_path = f'{save_dir}\\rmbg.jpg'
    
    cloth_img = PIL.Image.open(cloth_path)
    mask_img = PIL.Image.open(f'{save_dir}\\clothseg.png')
    file_path = ladivton.run_inference(id = id, cloth_img = cloth_img, mask_img = mask_img, reference_id = reference_id, index = index, save_dir = save_dir, INIT_VARS = LADIVTON_INIT_VARS, **LADIVTON_SETTINGS)
    return jsonify({'status': 'success', 'path': file_path, 'reference_id': reference_id ,'image': base64.b64encode(open(file_path, 'rb').read()).decode('utf-8')})
    
    """
    except Exception as e:
        return jsonify({'status': f'error : {str(e)}'}), 500
    """

@app.route("/cluster", methods=['POST'])
def clustering():
    #try:
    data = request.json
    print(f'cluster post body: {data}')
    
    id = data['id']
    category = data['category']
    is_upper = category == 'upper_body'
    img_path = f'.\\images\\runtime\\{id}\\rmbg.png'
    
    sorted_list = cluster.run_inference(is_upper, img_path, CLUSTER_INIT_VARS)
    cluster_id_list = sorted_list['sorted_images_list'].tolist()
    cluster_id_list = [i.split('_')[0] for i in cluster_id_list]
    
    print(f'cluster_id_list: {cluster_id_list}')
    
    with open(f'.\\images\\runtime\\{id}\\cluster_id_list.txt', 'w') as f:
        for item in cluster_id_list:
            f.write("%s\n" % item)
        
    return jsonify({'status': 'success', 'cluster_id_list': cluster_id_list})
    
    """except Exception as e:
        return jsonify({'status': f'error : {str(e)}'}), 500"""


@app.route("/clothseg", methods=['POST'])
def clothseg_predict():
    #try:
    data = request.json
    print(f'clothseg post body: {data}')
    
    id = data['id']
    img_path = f'.\\images\\runtime\\{id}\\rmbg.png'
    output_dir = f'.\\images\\runtime\\{id}'
    
    output_path = clothseg.run_inference(img_path, output_dir, CLOTHSEG_INIT_VARS)
    print(f'path: {output_path}')
    return jsonify({'status': 'success', 'path': output_path, 'image': base64.b64encode(open(output_path, 'rb').read()).decode('utf-8')})
    
    """except Exception as e:
        return jsonify({'status': f'error : {str(e)}'}), 500"""




def printLogo():
    print("""\
 ____   _____  ____  _      ___   ____            _     ___ 
|  _ \ | ____|/ ___|| |    / _ \ / ___|          / \   |_ _|
| |_) ||  _| | |    | |   | | | |\___ \  _____  / _ \   | | 
|  _ < | |___| |___ | |___| |_| | ___) ||_____|/ ___ \  | | 
|_| \_\|_____|\____||_____|\___/ |____/       /_/   \_\|___|
                                                             """)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
