import yaml

CONFIG_DEFAULT = "/home/sf123/ctMSNovelist/config.yaml"

config = {}


def load_config(path=CONFIG_DEFAULT):
    global config
    with open(path, "r") as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)
        config.update(config_)


load_config(CONFIG_DEFAULT)
