import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
project_name = "mlproject"
list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/Data_ingestion/__init__.py",
    f"src/{project_name}/components/Data_transformation/__init__.py",
    f"src/{project_name}/components/Model_trainer.py",
    f"src/{project_name}/components/MOdel_evaluation.py",
    f"src/{project_name}/piplines/__init__.py",
    f"src/{project_name}/piplines/trainging_pipline.py",
    f"src/{project_name}/piplines/prediction_pipline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Docker_file",
    "requirement.txt",
    "setup.py"
]
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating directorie {filedir} for the file {filename}")
    
    if(not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"creating empty file: {filepath}")
        
    else:
        logging.info(f"{filename} already exist")


