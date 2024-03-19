import json
from pathlib import Path


class DTUConfig:
    def __init__(self, args):
        self.settings = None
        args.add_argument('-c', '--config', default=None, type=str,
                          help='JSON config file (default: None)')
        args.add_argument('-i', '--input', default=None, type=str,
                          help='Input scan id (default: None)')
        args = args.parse_args()
        if hasattr(args, 'config') and args.config is not None:
            self.load_settings(args.config.strip())
        else:
            print("Configuration file need to be specified. Add '-c config.json', for example.")

        if hasattr(args, 'input') and args.input is not None and self.settings is not None:
            self.update_settings_with_scan_id(args.input)
        else:
            if self.settings is not None:
                self.settings["scan_id"] = ""

    def load_settings(self, file_name):
        try:
            with open(file_name, 'r') as openfile:
                self.settings = json.load(openfile)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {file_name}")
            self.settings = None

    def save_settings(self, file_name):
        try:
            with Path(file_name).open('wt') as handle:
                json.dump(self.settings, handle, indent=4, sort_keys=False)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}: {file_name}")

    def update_settings_with_scan_id(self, scan_id):
        """
        Given the base settings, use an individual scan id to add further directions and settings
        """
        base_dir = self.settings["base_dir"]
        image_dir = self.settings["image_dir"]

        # mode = settings["mode"]
        self.settings["scan_id"] = scan_id
        self.settings["scan_base_dir"] = f"{base_dir}{scan_id}/"
        self.settings["input_file"] = f"{image_dir}{scan_id}.nii.gz"
        # settings["segment_base_dir"] = f"{base_dir}{scan_id}/segmentations/"
        # settings["surface_dir"] = f"{base_dir}{scan_id}/surfaces/"
        # settings["landmark_dir"] = f"{base_dir}{scan_id}/landmarks/"
        # settings["centerline_dir"] = f"{base_dir}{scan_id}/centerlines/"
        # settings["statistics_dir"] = f"{base_dir}{scan_id}/statistics/"
        # settings["lock_file_base"] = f"{base_dir}{scan_id}_lock_"
        # is_cfa = settings.get("CFA", False)
        # if is_cfa:
        #     settings["visualization_dir"] = f"{base_dir}visualizations_{mode}_CFA/"
        # else:
        #     settings["visualization_dir"] = f"{base_dir}visualizations_{mode}/"
        #
        # settings["visualization_output"] = f"{settings['visualization_dir']}{scan_id}_visualization.png"
