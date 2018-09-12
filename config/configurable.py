import json

class Configurable(object):
    def __init__(self, section):
        self._config_filename = '/home/stevenwd/DRGD/config/config.json'
        # self._config_filename = '/Users/stevenwd/DRGD/config/config.json'
        self._config = json.load(open(self._config_filename, 'r'))
        self._section = section

    @property
    def config(self):
        return self._config[self._section]

    def get_config(self, section=None, key=None):
        if section == None:
            section = self._section
        return self._config[section][key]

    def update_config(self, key, value):
        self._config[self._section][key] = value

    def save_config(self):
        json.dump(self._config, open(self._config_filename, 'w'), indent=4, ensure_ascii=False)

    @property
    def base_dir(self):
        return self._config['base_dir']
