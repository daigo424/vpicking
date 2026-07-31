#!/usr/bin/env python3
"""edge/ros2_ws/src/ に、launchファイルだけを持つROS2パッケージを対話的に作成する。

nav2_bringup_custom, safety_bringup と同じ「コードは持たず、他パッケージの
ノード/launchファイルを起動するだけ」の薄いbringupパッケージを想定している。
colcon build後、他のlaunchファイルから
    get_package_share_directory('<新パッケージ名>')
経由でIncludeLaunchDescriptionできる状態に仕上げる
(package.xml/CMakeLists.txtでlaunch/をshare/へインストールするところまでやる)。

実行例:
    python3 scripts/create_launch_package.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROS2_SRC = REPO_ROOT / "edge" / "ros2_ws" / "src"

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

PACKAGE_XML_TEMPLATE = """<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{pkg_name}</name>
  <version>0.1.0</version>
  <description>
    {description}
  </description>
  <maintainer email="noreply@example.com">vpicking</maintainer>
  <license>Apache-2.0</license>
{exec_depends}
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
"""

CMAKELISTS_TEMPLATE = """cmake_minimum_required(VERSION 3.8)
project({pkg_name})

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  DESTINATION share/${{PROJECT_NAME}}
)

ament_package()
"""

LIFECYCLE_LAUNCH_BODY = '''from launch import LaunchDescription
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    {var}_cmd = LifecycleNode(
        package='{target_package}',
        executable='{executable}',
        name='{node_name}',
        namespace='',
        output='screen',
    )

    ld = LaunchDescription()
    ld.add_action({var}_cmd)
    return ld
'''

NODE_LAUNCH_BODY = '''from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    {var}_cmd = Node(
        package='{target_package}',
        executable='{executable}',
        name='{node_name}',
        namespace='',
        output='screen',
    )

    ld = LaunchDescription()
    ld.add_action({var}_cmd)
    return ld
'''

INCLUDE_LAUNCH_BODY = '''import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    {var}_dir = get_package_share_directory('{target_package}')

    {var}_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join({var}_dir, 'launch', '{target_launch_file}')
        )
    )

    ld = LaunchDescription()
    ld.add_action({var}_cmd)
    return ld
'''

EMPTY_LAUNCH_BODY = '''from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()
    return ld
'''


def find_packages() -> list[Path]:
    return sorted(
        {p.parent for p in ROS2_SRC.rglob("package.xml")},
        key=lambda p: p.relative_to(ROS2_SRC).as_posix(),
    )


def find_group_dirs() -> dict[str, Path]:
    """package.xmlを直接持たない、パッケージの入れ物になっているディレクトリ(safety/等)を集める。"""
    groups: dict[str, Path] = {".": ROS2_SRC}
    for child in sorted(ROS2_SRC.iterdir()):
        if child.is_dir() and not (child / "package.xml").exists():
            groups[child.name] = child
    return groups


def package_name_from_xml(pkg_dir: Path) -> str:
    text = (pkg_dir / "package.xml").read_text(encoding="utf-8")
    match = re.search(r"<name>\s*([^<\s]+)\s*</name>", text)
    return match.group(1) if match else pkg_dir.name


def all_known_package_names() -> set[str]:
    return {package_name_from_xml(p) for p in find_packages()}


def render_launch_file(docstring: str, body: str) -> str:
    header = "#!/usr/bin/env python3\n"
    if docstring:
        header += f'"""{docstring}"""\n'
    return header + body


def prompt(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{question}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("空欄にはできません。")


def prompt_optional(question: str) -> str:
    return input(f"{question} (空欄可): ").strip()


def prompt_choice(question: str, options: list[tuple[str, str]], default_key: str) -> str:
    print(question)
    for key, label in options:
        marker = "*" if key == default_key else " "
        print(f"  [{marker}] {key}) {label}")
    while True:
        answer = input(f"選択 [{default_key}]: ").strip() or default_key
        if answer in {key for key, _ in options}:
            return answer
        print("選択肢の中から入力してください。")


def prompt_new_package_name(existing_names: set[str]) -> str:
    while True:
        name = prompt("新しいパッケージ名")
        if not NAME_RE.match(name):
            print("小文字英数字とアンダースコアのみ、先頭は英字にしてください。")
            continue
        if name in existing_names:
            print(f"'{name}' は既に edge/ros2_ws/src 配下に存在します。別の名前にしてください。")
            continue
        return name


def prompt_location() -> Path:
    groups = find_group_dirs()
    keys = list(groups.keys())
    print("配置先を選んでください:")
    for key in keys:
        rel = groups[key].relative_to(REPO_ROOT).as_posix()
        label = "ルート直下(nav2_bringup_customと同じ階層)" if key == "." else f"{rel}/ 配下"
        print(f"  {key}) {label}")
    print("  (上記以外の相対パスを直接入力することもできます。例: safety/新グループ)")
    while True:
        answer = input(f"配置先 [.]: ").strip() or "."
        if answer in groups:
            return groups[answer]
        # 自由入力パス
        custom = ROS2_SRC / answer
        return custom


def prompt_target_package(question: str) -> str:
    packages = find_packages()
    print(question)
    for i, pkg_dir in enumerate(packages, start=1):
        rel = pkg_dir.relative_to(ROS2_SRC).as_posix()
        print(f"  {i}) {rel}")
    print("  0) このリポジトリに無いパッケージ名を直接入力する(例: nav2_bringup, turtlebot3_gazebo)")
    while True:
        raw = input(f"番号を入力 [0-{len(packages)}]: ").strip()
        if raw == "0":
            return prompt("パッケージ名")
        if raw.isdigit() and 1 <= int(raw) <= len(packages):
            return package_name_from_xml(packages[int(raw) - 1])
        print("番号が不正です。")


def sanitize_var(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower()) or "target"


def build_exec_depends_block(deps: list[str]) -> str:
    if not deps:
        return ""
    lines = "\n".join(f"  <exec_depend>{dep}</exec_depend>" for dep in deps)
    return lines + "\n"


def main() -> None:
    print("=== ROS2 launch専用パッケージ作成 ===")

    existing_names = all_known_package_names()
    pkg_name = prompt_new_package_name(existing_names)
    location = prompt_location()
    description = prompt("パッケージの説明(1行)", default=f"vpicking: {pkg_name} bringupパッケージ")

    kind = prompt_choice(
        "最初のlaunchファイルの内容は?",
        [
            ("node", "既存パッケージの1ノードを起動する(LifecycleNode)"),
            ("plain_node", "既存パッケージの1ノードを起動する(通常のNode)"),
            ("include", "既存パッケージのlaunchファイルをincludeするだけの薄いラッパー"),
            ("empty", "空のlaunchファイル(あとで自分で書く)"),
        ],
        default_key="node",
    )

    exec_depends: list[str] = []

    if kind in ("node", "plain_node"):
        target_package = prompt_target_package("起動するノードが属するパッケージを選んでください:")
        executable = prompt("実行ファイル名", default=f"{target_package}_node")
        node_name = prompt("ノード名(name=)", default=target_package)
        var = sanitize_var(node_name)
        body_template = LIFECYCLE_LAUNCH_BODY if kind == "node" else NODE_LAUNCH_BODY
        docstring = prompt_optional(f"launchファイルの説明(1行、例: {target_package}ノードを起動する。)")
        launch_content = render_launch_file(
            docstring,
            body_template.format(
                var=var,
                target_package=target_package,
                executable=executable,
                node_name=node_name,
            ),
        )
        exec_depends.append(target_package)
    elif kind == "include":
        target_package = prompt_target_package("includeするlaunchファイルが属するパッケージを選んでください:")
        target_launch_file = prompt("includeするlaunchファイル名", default=f"{target_package}.launch.py")
        var = sanitize_var(target_package)
        docstring = prompt_optional(
            f"launchファイルの説明(1行、例: {target_package} パッケージの {target_launch_file} をincludeする薄いラッパー。)"
        )
        launch_content = render_launch_file(
            docstring,
            INCLUDE_LAUNCH_BODY.format(
                var=var,
                target_package=target_package,
                target_launch_file=target_launch_file,
            ),
        )
        exec_depends.append(target_package)
    else:
        docstring = prompt_optional("launchファイルの説明(1行)")
        launch_content = render_launch_file(docstring, EMPTY_LAUNCH_BODY)

    extra_deps_raw = prompt_optional("追加のexec_depend(カンマ区切り)")
    if extra_deps_raw:
        exec_depends.extend(d.strip() for d in extra_deps_raw.split(",") if d.strip())

    launch_file_name = prompt("launchファイル名(拡張子なし)", default=pkg_name)

    pkg_dir = location / pkg_name
    if pkg_dir.exists():
        print(f"[エラー] {pkg_dir.relative_to(REPO_ROOT)} は既に存在します。", file=sys.stderr)
        sys.exit(1)

    launch_dir = pkg_dir / "launch"
    launch_dir.mkdir(parents=True)

    (pkg_dir / "package.xml").write_text(
        PACKAGE_XML_TEMPLATE.format(
            pkg_name=pkg_name,
            description=description,
            exec_depends=build_exec_depends_block(exec_depends),
        ),
        encoding="utf-8",
    )
    (pkg_dir / "CMakeLists.txt").write_text(
        CMAKELISTS_TEMPLATE.format(pkg_name=pkg_name),
        encoding="utf-8",
    )
    launch_path = launch_dir / f"{launch_file_name}.launch.py"
    launch_path.write_text(launch_content, encoding="utf-8")
    launch_path.chmod(0o755)

    print()
    print(f"作成しました: {pkg_dir.relative_to(REPO_ROOT)}/")
    print(f"  - package.xml")
    print(f"  - CMakeLists.txt")
    print(f"  - launch/{launch_file_name}.launch.py")

    print()
    print("次の手順:")
    print(f"  make build PKG={pkg_name}")
    print()
    print("他のlaunchファイルからIncludeLaunchDescriptionで呼び出す例:")
    print(f"""
    {pkg_name}_dir = get_package_share_directory('{pkg_name}')
    {pkg_name}_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join({pkg_name}_dir, 'launch', '{launch_file_name}.launch.py')
        )
    )
    ld.add_action({pkg_name}_cmd)
""")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
        print("中断しました。")
        sys.exit(130)
