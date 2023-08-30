from setuptools import find_packages,setup
from typing import List


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."

__version__ = "0.0.0"

REPO_NAME = "income"
AUTHOR_USER_NAME = "Milind-Shende"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "milind.shende24@rediffmail.com"

def get_requirements()->List[str]:

    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace("\n","") for requirement_name in requirement_list]

    #Removing HYPHEN _E_DOT
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list



setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements(),
    )