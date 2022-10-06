
from datetime import datetime
import os
import shutil

class FileUtils:
    def __init__(self):
        pass

    def make_dir(self, *subpaths):
        path = self.abs_path(*subpaths)
        os.makedirs(path, exist_ok=True)
        return path

    def abs_path(self, *subpaths):
        return self.abs_path_with_splits(self, None, *subpaths)

    def abs_path_with_splits(self, number_of_splits=None, *subpaths):
        path = os.path.join(*subpaths)
        if os.path.isabs(path):
            return path
        if '__file__' in globals():
            parts = os.path.split(__file__)[:-1]

            if number_of_splits is not None:
                p = parts

                for i in range(number_of_splits):
                    p = p[0]
                    p = os.path.split(p)

                parts = p[:-1]

            parts += subpaths

            return os.path.join(*parts)
        return os.path.abspath(*subpaths)

    def clear_dir_and_files(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))

    def create_dirs(self, *sub_paths):
        dir_path = self.abs_path_with_splits(1, *sub_paths)

        if os.path.isdir(dir_path) is False:
            os.makedirs(dir_path)

        return dir_path

    def create_sub_dir(self, dir_path, label_name):
        sub_dir = os.path.join(dir_path, label_name)

        if os.path.isdir(sub_dir) is False:
            os.mkdir(sub_dir)
        else:
            self.clear_dir_and_files(sub_dir)
            print("[{}] directory already exists.".format(sub_dir))

        return sub_dir

    def create_path_dir(self, label_name, *sub_paths):
        dir_path = self.create_dirs(*sub_paths)
        sub_dir = self.create_sub_dir(dir_path, label_name)
        return sub_dir

    def list_files(self, *subpaths, paths_only=False):
        dir_path = self.abs_path(*subpaths)
        res = []
        for f in os.listdir(dir_path):
            path = os.path.join(dir_path, f)
            if os.path.isfile(path):
                if paths_only:
                    res.append(path)
                else:
                    res.append((path, f))
        return res

    def clear_dir(self, *subpaths):
        dir_path = self.abs_path(*subpaths)
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    def clear_or_make_dir(self, *subpaths):
        dir_path = self.abs_path(*subpaths)
        if os.path.exists(dir_path):
            self.clear_dir(dir_path)
        else:
            self.make_dir(dir_path)

    def copy_files(self, from_path, to_path, clear_first=True, move=False, rename_cb=None):
        from_path = self.abs_path(from_path) if type(from_path) == str else self.abs_path(*from_path)
        to_path = self.abs_path(to_path) if type(to_path) == str else self.abs_path(*to_path)
        if os.path.exists(to_path):
            if clear_first:
                self.clear_dir(to_path)
        else:
            self.make_dir(to_path)
        for from_file_path, from_file_name in self.list_files(from_path, paths_only=True):
            to_file_name = rename_cb(from_file_name) if rename_cb else from_file_name
            to_file_path = os.path.join(to_path, to_file_name)
            shutil.copy(from_file_path, to_file_path)

    def get_path(self, *sub_paths):
        dir_path = self.abs_path_with_splits(1, *sub_paths)
        print("[{}] directory exists.".format(dir_path))
        return dir_path

    def cur_datetime(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def rename(self, old_name, new_name):
        os.rename(old_name, new_name)





