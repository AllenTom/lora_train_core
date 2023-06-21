import os.path
import shutil
from abc import abstractmethod
import hashlib

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


rules = [ViTCheckRule(), BLIPCheckRule()]


def check():
    for rule in rules:
        if not rule.check():
            print(rule.get_message())
            if rule.get_message_level() == -1:
                exit(1)
    print("检查通过")
