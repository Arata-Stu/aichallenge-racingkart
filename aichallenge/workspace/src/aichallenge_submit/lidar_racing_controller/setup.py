import os
from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "lidar_racing_controller"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{PACKAGE_NAME}"],
        ),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
        (os.path.join("share", PACKAGE_NAME, "config"), glob("config/*.yaml")),
        (os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*.xml")),
        (os.path.join("share", PACKAGE_NAME, "models"), glob("models/*")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=False,
    maintainer="Arata Tanaka",
    maintainer_email="tanaka.arata.y5@s.gifu-u.ac.jp",
    description="LiDAR-only PyTorch policy controller for the AI Challenge racing kart",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "lidar_racing_controller_node = lidar_racing_controller.node:main",
        ],
    },
)
