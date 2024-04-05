from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    '''
    this functions will return list of requirements

    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n',"")for req in requirements]
    return requirements
setup(
name='Machine Learning Project for Credit Card Fraud Detection',
author_email='hassanraza.rashid5@gmail.com',
author='CH Hassan Raza',
version='0.0.1',
packages=find_packages(),
install_requires=get_requirements('requirement.txt'),
)









