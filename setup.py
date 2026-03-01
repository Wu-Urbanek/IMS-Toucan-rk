from setuptools import setup, find_packages

setup(
    name="ims_toucan",
    version="0.1",
    packages=find_packages(), # 這會自動把所有含 __init__.py 的資料夾打包
)