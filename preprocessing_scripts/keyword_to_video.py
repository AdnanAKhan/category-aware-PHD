import os
import json


class VideoToKeyWordMapping:

    def __init__(self):
        """
        initiallization of data directories
        """
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
        self.PROCESSED_KEYWORDS_DIR = os.path.join(self.DATA_DIR, 'metadata', 'processed_keywords')
        self.GLOBAL_MAPPING = {}

    def process_single_file(self, file_name):
        keywords = json.load(open(os.path.join(self.PROCESSED_KEYWORDS_DIR, file_name), 'r'))

        for word in keywords:
            if word not in self.GLOBAL_MAPPING:
                self.GLOBAL_MAPPING[word] = [file_name]
            else:
                self.GLOBAL_MAPPING[word].append(file_name)

    def process(self):
        """
        process all files
        :return:
        """
        for file_name in os.listdir(self.PROCESSED_KEYWORDS_DIR):
            self.process_single_file(file_name)

        with open(os.path.join(self.DATA_DIR, 'keywords_to_video_mapping.json'), 'w') as f:
            json.dump(self.GLOBAL_MAPPING, f)

    def count(self):
        count_dict = {}
        for k, v in self.GLOBAL_MAPPING.items():
            count_dict[k] = len(v)

        sorted_dict = dict(sorted(count_dict.items(), key=lambda kv: kv[1], reverse=True))
        with open(os.path.join(self.DATA_DIR, 'keywords_to_video_mapping_sorted.json'), 'w') as f:
            json.dump(sorted_dict, f)


if __name__ == '__main__':
    obj = VideoToKeyWordMapping()
    obj.process()
    obj.count()
