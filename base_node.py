NODE_NAME = "Akatz Seamless Tiling"
NODE_POSTFIX = "| Akatz"


class BaseNode:
    CATEGORY = NODE_NAME


class ImageNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Image"


class LatentNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Latent"


class LogicNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Logic"


class MaskNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Mask"
