import json
import os.path
import re
import shutil
from abc import abstractmethod
import hashlib

import pkg_resources

from modules import share


# info level -1=Fatal 0=error, 1=warning, 2=info, 3=debug
class CheckRule:
    @abstractmethod
    def check(self) -> bool:
        pass

    @abstractmethod
    def get_message(self) -> str:
        pass

    @abstractmethod
    def get_message_level(self) -> int:
        pass


def check_dependencies(dependencies):
    missing_deps = []
    for dep in dependencies:
        regex = r"^(\w+)==(\d+\.\d+\.\d+)$"
        match = re.match(regex, dep)
        if match:
            package_name = match.group(1)
            version = match.group(2)
            # print("Package name:", package_name)
            # print("Version:", version)
            try:
                dist = pkg_resources.get_distribution(package_name)
                if (dist.version != version):
                    # print("Version mismatch:", dist.version, "!=", version)
                    missing_deps.append(package_name)

            except Exception:
                # print("Package not found:", package_name)
                missing_deps.append(package_name)

    return missing_deps


class DepRule(CheckRule):
    def __init__(self):
        self.message = ""
        self.level = 2

    def check(self) -> bool:
        with open('requirements.txt', 'r') as file:
            requirements = file.read().splitlines()

        # Check if dependencies are installed
        missing_dependencies = check_dependencies(requirements)
        if missing_dependencies:
            # print("The following dependencies are missing:")
            for dep in missing_dependencies:
                # print(dep)
                pass
        else:
            # print("All dependencies are installed.")
            pass
        self.message = "缺少依赖项: " + ",".join(missing_dependencies)
        return len(missing_dependencies) == 0

    def get_message(self) -> str:
        return self.message

    def get_message_level(self) -> int:
        return -1


class ViTCheckRule(CheckRule):

    def __init__(self):
        self.message = ""
        self.level = 2

    def check(self) -> bool:
        if not os.path.exists(share.vit_model_path):
            self.message = "ViT 模型未找到，使用时需要下载"
            self.level = 1
            return False
        # check hash
        with open(share.vit_model_path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            if md5 != share.vit_model_hash:
                shutil.rmtree(share.vit_model_path)
                self.level = 1
                self.message = "ViT 模型文件MD5校验失败，已删除,请重新下载或运行时自动下载"
                return False
        return True

    def get_message(self) -> str:
        return self.message

    def get_message_level(self) -> int:
        return 1


class BLIPCheckRule(CheckRule):

    def __init__(self):
        self.message = ""
        self.level = 2

    def check(self) -> bool:
        if not os.path.exists(share.blip_model_path):
            self.message = "BLIP 模型未找到，使用时需要下载"
            self.level = 1
            return False
        # check hash
        with open(share.blip_model_path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            if md5 != share.blip_model_hash:
                shutil.rmtree(share.blip_model_path)
                self.level = 1
                self.message = "BLIP 模型文件MD5校验失败，已删除, 请重新下载或运行时自动下载"
                return False
        return True

    def get_message(self) -> str:
        return self.message

    def get_message_level(self) -> int:
        return 1


def get_rules(case="all"):
    if case == "all":
        return [DepRule(), ViTCheckRule(), BLIPCheckRule()]
    elif case == "start":
        return [DepRule()]
    else:
        return []


def check(json_out=False, case="all"):
    for rule in get_rules(case):
        if not rule.check():
            # if (json_out):
            #     out_obj = {
            #         "message": rule.get_message(),
            #         "level": rule.get_message_level(),
            #         "event": "message"
            #
            #     }
            #     print(json.dumps(out_obj, ensure_ascii=False),end="\n")
            # else:
            #     print(rule.get_message(),end="\n")
            if rule.get_message_level() == -1:
                print(json.dumps({
                    "message": "检查未通过",
                    "level": -1,
                    "event": "checkFailed"
                },ensure_ascii=False),end="\n")
                exit(1)
    if (json_out):
        out_obj = {
            "message": "检查通过",
            "level": 2,
            "event": "checkPassed"
        }
        print(json.dumps(out_obj, ensure_ascii=False),end="\n")
