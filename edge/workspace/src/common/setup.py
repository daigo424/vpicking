from setuptools import find_packages, setup

package_name = "common"

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
    description="vision_pickingパッケージのROS2ノードと、colcon管理外のscripts/配下のツールの両方から使う共通処理",
    license="Apache-2.0",
    tests_require=["pytest"],
)
