from pytorch_lightning.utilities.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--onnx.path", default="export/model.onnx")
        parser.add_argument("--torchscript.path", default='export/model.torchscript')
