import json
import yaml

import subprocess
from os import listdir, system
from os.path import isfile, join

import re

def list_files_in_dir(path: str = "../src/template_pipelines/config/") -> list[str]:
    return [f for f in listdir(path) if isfile(join(path, f))]

def load_json(index_path: str = "../index.json") -> dict:
    index_file = open(file=index_path, mode="r")
    return json.load(fp=index_file)

def open_new_index_file(index_path: str = "index.json"):
    new_index_path = index_path[:-5] + "_new" + index_path[-5:]
    return open(file=new_index_path, mode="w")

def verify_filename(file: str) -> bool:
    return file.startswith("pipeline_") and file.endswith(".yml")

def load_config_data(file: str, path: str = "../src/template_pipelines/config/") -> dict:
    config_path = path + file
    config_file = open(file=config_path, mode="r")
    return yaml.safe_load(stream=config_file)

def get_platforms(data: dict) -> set:
    if "platforms" in data.keys(): config_platforms = set(data["platforms"])
    else: config_platforms = set()
    return config_platforms

def extract_pipeline_name(file: str) -> str:
    return file.removeprefix("pipeline_").removesuffix(".yml")

def parse_pipeline_config(pipelines_index: dict, pipeline_name: str, config_platforms: dict) -> tuple[list, list]:
    missing_info, missmatch_info = [], []
    if pipeline_name in pipelines_index.keys():
        pipeline_index_data = pipelines_index[pipeline_name]
        pipeline_index_platforms = get_platforms(data=pipeline_index_data)
        if not (config_platforms == pipeline_index_platforms):
            platform_set = config_platforms.union(pipeline_index_platforms)
            log_info = (pipeline_name, config_platforms, pipeline_index_platforms, list(platform_set))
            missmatch_info = [log_info]
    else:
        print(f"Please complete information about {pipeline_name} in index.json")
        missing_info = [pipeline_name]
    return missing_info, missmatch_info

def verify_file(file: str, missing_pipelines: list, platform_missmatching_pipelines: list, index_json: dict, pipelines_index: dict, path: str = "../src/template_pipelines/config/") -> tuple[list, list, dict]:
    if verify_filename(file):
        config_data = load_config_data(file=file, path=path)
        new_config_path = path + "new_" + file
        config_platforms = get_platforms(data=config_data)
        pipeline_name = extract_pipeline_name(file)

        missing_info, missmatch_info = parse_pipeline_config(
            pipelines_index=pipelines_index,
            pipeline_name=pipeline_name,
            config_platforms=config_platforms
        )

        missing_pipelines.extend(missing_info)
        platform_missmatching_pipelines.extend(missmatch_info)

        if len(missmatch_info) == 4:
            platform_missmatching_pipelines.extend(missmatch_info)
            config_data["platforms"] = missmatch_info[3]
            index_json["pipelines"][pipeline_name]["platforms"] = missmatch_info[3]
            new_config_file = open(file=new_config_path, mode="w")
            yaml.safe_dump(data=config_data, stream=new_config_file)

    return missing_pipelines, platform_missmatching_pipelines, index_json

def verify_files(files_list: list[str], index_json: dict, path: str = "../src/template_pipelines/config/") -> list[str]:
    pipelines_index = index_json["pipelines"]
    missing_pipelines, platform_missmatching_pipelines = [], []
    for file in files_list:
        missing_pipelines, platform_missmatching_pipelines, index_json = verify_file(
            file=file,
            missing_pipelines=missing_pipelines,
            platform_missmatching_pipelines=platform_missmatching_pipelines,
            index_json=index_json,
            pipelines_index=pipelines_index,
            path=path
        )
    
    print(f"Missing Platform: {missing_pipelines}")
    print(f"Platform Missmatching: {platform_missmatching_pipelines}")

    return missing_pipelines, index_json

def validate_prefix_suffix(filename: str, prefix_suffix_list: list[str], option: str = "prefix") -> bool:
    if option == "prefix": method = lambda x, y: x.startswith(y)
    else: method = lambda x, y: x.endswith(y)

    retval = False
    for prefix_suffix in prefix_suffix_list:
        retval = (retval or method(filename, prefix_suffix))
    
    return retval

def get_files(path: str, pipeline: str, prefixes: list[str], suffixes: list[str] = [".json", ".yml", ".yaml"]) -> list[str]:
    files_list = list_files_in_dir(path=path)
    config_files_list = []
    for filename in files_list:
        if validate_prefix_suffix(filename=filename, prefix_suffix_list=prefixes) and \
            validate_prefix_suffix(filename=filename, prefix_suffix_list=suffixes, option="suffix") and\
            pipeline in filename:
            config_files_list.append(filename)
    return config_files_list

def get_config_files(path: str, pipeline: str) -> list[str]:
    return get_files(path=path, pipeline=pipeline, prefixes=["config_", "model_"])

def get_readme_files(path: str, pipeline: str) -> list[str]:
    get_files(path=path, pipeline=pipeline, prefixes=["readme_"])

def get_reaquirements_files(path: str, pipeline: str) -> list[str]:
    get_files(path=path, pipeline=pipeline, prefixes=["requirements_"])

def get_uri(pipeline: str) -> str:
    return f"https://github.com/procter-gamble/de-cf-pyrogai-op-pipelines/blob/main/index.json?pipeline-{pipeline}"

def create_pipeline_data(pipeline: str, pipeline_name: str, path: str = "../src/template_pipelines/config/"):
    pipeline_data = load_config_data(file=pipeline_name, path=path)
    description = pipeline_data["description"]
    pipeline_index_data = dict()
    pipeline_index_data["config_files"] = get_config_files(path=path, pipeline=pipeline)
    pipeline_index_data["desc"] = description
    pipeline_index_data["guid"] = ""
    pipeline_index_data["include_utils"] = True
    pipeline_index_data["name"] = pipeline
    pipeline_index_data["path"] = path + pipeline_name
    pipeline_index_data["platforms"] = list(get_platforms(pipeline_data))
    pipeline_index_data["readme_files"] = get_readme_files(path="../src/template_pipelines/", pipeline=pipeline)
    pipeline_index_data["requirements_files"] = get_reaquirements_files(path="../src/template_pipelines/reqs/", pipeline=pipeline)
    pipeline_index_data["runtimes_mapping"] = None
    pipeline_index_data["tags"] = []
    pipeline_index_data["uri"] = get_uri(pipeline=pipeline)
    pipeline_index_data["utils_paths"] = None
    return pipeline_index_data

def run_tp_command():
    result = subprocess.run(["tp", "validate"], stdout=subprocess.PIPE)
    return result.stdout.decode(encoding="utf-8")

def extract_regex(message: str, extracted) -> str:
    return message[extracted.start():extracted.start()+extracted.end()]

def extract_guid(message: str) -> str:
    message.replace("-", "!")
    message.replace(".", "!")
    extracted = re.search("Pipeline \"quickstart\" has an incorrect value for the GUID field, expected", message)
    return message[extracted.end()+1:extracted.end()+37] 

def validate_index_json() -> str:
    tp_message = run_tp_command()
    guid = None
    if tp_message.startswith("Index is valid"): 
        print("index.json is valid")
    else: 
        print("index.json validatoin failed")
        guid = extract_guid(tp_message)
    print(guid)
    return guid

def complete_guid(pipeline: str) -> None:
    guid = validate_index_json()

    if guid is not None:
        index_json[pipeline]["guid"] = guid
        new_index_file = open_new_index_file(index_path=index_path)
        json.dump(obj=index_json, fp=new_index_file, indent="  ")
        new_index_file.close()

def complete_missing_pipelines(missing_pipelines: list, index_json: dict, path: str = "../src/template_pipelines/config/") -> None:
    for pipeline in missing_pipelines:
        pipeline_name = "pipeline_" + pipeline + ".yml"
        
        index_json[pipeline] = create_pipeline_data(
            pipeline=pipeline,
            pipeline_name=pipeline_name,
            path=path
        )

        new_index_file = open_new_index_file(index_path=index_path)
        json.dump(obj=index_json, fp=new_index_file, indent="  ")
        new_index_file.close()

        complete_guid(pipeline=pipeline)
        

if __name__ == "__main__":

    path = "../src/template_pipelines/config/"
    index_path = "../index.json"

    files_list = list_files_in_dir(path=path)
    index_json = load_json(index_path=index_path)
    
    missing_pipelines, index_json = verify_files(
        files_list=files_list, 
        index_json=index_json, 
        path=path
    )

    complete_missing_pipelines(
        missing_pipelines=missing_pipelines, 
        index_json=index_json, 
        path=path
    )
