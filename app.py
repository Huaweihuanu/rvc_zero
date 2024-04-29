import os
import gradio as gr
import spaces
from infer_rvc_python import BaseLoader
import random
import logging
import time
import soundfile as sf
from infer_rvc_python.main import download_manager
import zipfile

logging.getLogger("infer_rvc_python").setLevel(logging.ERROR)

converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)

title = "<center><strong><font size='7'>RVC⚡ZERO</font></strong></center>"
description = "This demo is provided for educational and research purposes only. The authors and contributors of this project do not endorse or encourage any misuse or unethical use of this software. Any use of this software for purposes other than those intended is solely at the user's own risk. The authors and contributors shall not be held responsible for any damages or liabilities arising from the use of this demo inappropriately."
theme = "aliabid94/new-theme"

PITCH_ALGO_OPT = [
    "pm",
    "harvest",
    "crepe",
    "rmvpe",
    "rmvpe+",
]


def find_files(directory):
    file_paths = []
    for filename in os.listdir(directory):
        # Check if the file has the desired extension
        if filename.endswith('.pth') or filename.endswith('.zip') or filename.endswith('.index'):
            # If yes, add the file path to the list
            file_paths.append(os.path.join(directory, filename))

    return file_paths


def unzip_in_folder(my_zip, my_dir):
    with zipfile.ZipFile(my_zip) as zip:
        for zip_info in zip.infolist():
            if zip_info.is_dir():
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, my_dir)


def find_my_model(a_, b_):

    if a_ is None or a_.endswith(".pth"):
        return a_, b_

    txt_files = []
    for base_file in [a_, b_]:
        if base_file is not None and base_file.endswith(".txt"):
            txt_files.append(base_file)
    
    directory = os.path.dirname(a_)
    
    for txt in txt_files:
        with open(txt, 'r') as file:
            first_line = file.readline()
    
        download_manager(
            url=first_line.strip(),
            path=directory,
            extension="",
        )
    
    for f in find_files(directory):
        if f.endswith(".zip"):
            unzip_in_folder(f, directory)
    
    model = None
    index = None
    end_files = find_files(directory)
    
    for ff in end_files:
        if ff.endswith(".pth"):
            model = os.path.join(directory, ff)
            gr.Info(f"Model found: {ff}")
        if ff.endswith(".index"):
            index = os.path.join(directory, ff)
            gr.Info(f"Index found: {ff}")

    if not model:
        gr.Error(f"Model not found in: {end_files}")
    
    if not index:
        gr.Warning("Index not found")
    
    return model, index


@spaces.GPU()
def convert_now(audio_files, random_tag, converter):
    return converter(
        audio_files,
        random_tag,
        overwrite=False,
        parallel_workers=4
    )


def run(
    audio_files,
    file_m,
    pitch_alg,
    pitch_lvl,
    file_index,
    index_inf,
    r_m_f,
    e_r,
    c_b_p,
):
    if not audio_files:
        raise ValueError("The audio pls")
    
    if isinstance(audio_files, str):
        audio_files = [audio_files]

    if file_m is not None and file_m.endswith(".txt"):
        file_m, file_index = find_my_model(file_m, file_index)
        print(file_m, file_index)

    random_tag = "USER_"+str(random.randint(10000000, 99999999))

    converter.apply_conf(
        tag=random_tag,
        file_model=file_m,
        pitch_algo=pitch_alg,
        pitch_lvl=pitch_lvl,
        file_index=file_index,
        index_influence=index_inf,
        respiration_median_filtering=r_m_f,
        envelope_ratio=e_r,
        consonant_breath_protection=c_b_p,
        resample_sr=44100 if audio_files[0].endswith('.mp3') else 0, 
    )
    time.sleep(0.3)

    return convert_now(audio_files, random_tag, converter)


def audio_conf():
    return gr.File(
        label="Audio files",
        file_count="multiple",
        type="filepath",
        container=True,
    )


def model_conf():
    return gr.File(
        label="Model file",
        type="filepath",
        height=130,
    )


def pitch_algo_conf():
    return gr.Dropdown(
        PITCH_ALGO_OPT,
        value=PITCH_ALGO_OPT[4],
        label="Pitch algorithm",
        visible=True,
        interactive=True,
    )


def pitch_lvl_conf():
    return gr.Slider(
        label="Pitch level",
        minimum=-24,
        maximum=24,
        step=1,
        value=0,
        visible=True,
        interactive=True,
    )


def index_conf():
    return gr.File(
        label="Index file",
        type="filepath",
        height=130,
    )


def index_inf_conf():
    return gr.Slider(
        minimum=0,
        maximum=1,
        label="Index influence",
        value=0.75,
    )


def respiration_filter_conf():
    return gr.Slider(
        minimum=0,
        maximum=7,
        label="Respiration median filtering",
        value=3,
        step=1,
        interactive=True,
    )


def envelope_ratio_conf():
    return gr.Slider(
        minimum=0,
        maximum=1,
        label="Envelope ratio",
        value=0.25,
        interactive=True,
    )


def consonant_protec_conf():
    return gr.Slider(
        minimum=0,
        maximum=0.5,
        label="Consonant breath protection",
        value=0.5,
        interactive=True,
    )


def button_conf():
    return gr.Button(
        "Inference",
        variant="primary",
    )


def output_conf():
    return gr.File(
        label="Result",
        file_count="multiple",
        interactive=False,
    )


def get_gui(theme):
    with gr.Blocks(theme=theme) as app:
        gr.Markdown(title)
        gr.Markdown(description)

        aud = audio_conf()
        with gr.Column():
            with gr.Row():
                model = model_conf()
                indx = index_conf()
        algo = pitch_algo_conf()
        algo_lvl = pitch_lvl_conf()
        indx_inf = index_inf_conf()
        res_fc = respiration_filter_conf()
        envel_r = envelope_ratio_conf()
        const = consonant_protec_conf()
        button_base = button_conf()
        output_base = output_conf()

        button_base.click(
            run,
            inputs=[
                aud,
                model,
                algo,
                algo_lvl,
                indx,
                indx_inf,
                res_fc,
                envel_r,
                const,
            ],
            outputs=[output_base],
        )

        
        gr.Examples(
            examples=[
                [
                    ["./test.ogg"],
                    "./model.pth",
                    "rmvpe+",
                    0,
                    "./model.index",
                    0.75,
                    3,
                    0.25,
                    0.50,
                ],
                [
                    ["./example2/test2.ogg"],
                    "./example2/model_link.txt",
                    "rmvpe+",
                    0,
                    "./example2/index_link.txt",
                    0.75,
                    3,
                    0.25,
                    0.50,
                ],
                [
                    ["./example3/test3.wav"],
                    "./example3/zip_link.txt",
                    "rmvpe+",
                    0,
                    None,
                    0.75,
                    3,
                    0.25,
                    0.50,
                ],
                
            ],
            fn=run,
            inputs=[
                aud,
                model,
                algo,
                algo_lvl,
                indx,
                indx_inf,
                res_fc,
                envel_r,
                const,
            ],
            outputs=[output_base],
            cache_examples=False,
        )

    return app


if __name__ == "__main__":

    app = get_gui(theme)

    app.queue(default_concurrency_limit=40)

    app.launch(
        max_threads=40,
        share=False,
        show_error=True,
        quiet=False,
        debug=False,
    )
