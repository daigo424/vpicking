from setuptools import find_packages, setup

package_name = "vision_picking"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vpicking",
    maintainer_email="noreply@example.com",
    description="Isaac Sim + ROS 2 + MoveIt 2によるビジョンピッキングの認識・制御ノード群",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gt_tf_publisher_node = vision_picking.gt_tf_publisher_node:main",
        ],
    },
)
