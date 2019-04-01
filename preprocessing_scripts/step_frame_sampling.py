import pickle
import os
import subprocess


class VideoFrameSampling:

    def __init__(self):
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
        self.SEGMENT_INFO_DIR = os.path.join(self.DATA_DIR, 'segment_info')
        self.SEGMENT_FRAME_DIR = os.path.join(self.DATA_DIR, 'video_segments')
        self.YOUTUBE_VIDEO_DIR = os.path.join(self.DATA_DIR)
        self.PARAMS = {
            'fps': 1,
            'size': 320
        }

    def process(self):
        filenames = os.listdir(self.SEGMENT_INFO_DIR)
        for filename in filenames:
            datum = pickle.load(open(os.path.join(self.SEGMENT_INFO_DIR, filename), "rb"))
            youtube_id = filename.split('_')[0]
            dest_dir = os.path.join(self.SEGMENT_FRAME_DIR, filename.split('.')[0])
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                print('working on {}'.format(filename))
                source_video_path = os.path.join(self.YOUTUBE_VIDEO_DIR, 'video', '{}.mp4'.format(youtube_id))
                if not os.path.exists(source_video_path):
                    source_video_path = os.path.join(self.YOUTUBE_VIDEO_DIR, 'youtube_dl', '{}.mp4'.format(youtube_id))

                for ind, item in enumerate(datum['highlight_segment']):
                    start_time = round(item['start'], 3)
                    end_time = round(item['end'], 3)

                    save_path = os.path.join(dest_dir, 'highlight', '{}'.format(ind))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    output_format = os.path.join(save_path, 'frame%04d.jpg')

                    subprocess.call(['ffmpeg',
                                     '-v',
                                     '0',
                                     '-i',
                                     source_video_path,
                                     '-vf',
                                     "select='between(t,{},{})'".format(start_time, end_time),
                                     '-vf',
                                     "fps={}".format(self.PARAMS['fps']),
                                     '-s',
                                     '{}x{}'.format(self.PARAMS['size'], self.PARAMS['size']),
                                     '-vsync',
                                     '0',
                                     output_format,
                                     '-hide_banner'
                                     ])

                for ind, item in enumerate(datum['non_highlight_segment']):
                    start_time = round(item['start'], 3)
                    end_time = round(item['end'], 3)

                    save_path = os.path.join(dest_dir, 'non_highlight', '{}'.format(ind))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    output_format = os.path.join(save_path, 'frame%04d.jpg')

                    subprocess.call(['ffmpeg',
                                     '-v',
                                     '0',
                                     '-i',
                                     source_video_path,
                                     '-vf',
                                     "select='between(t,{},{})'".format(start_time, end_time),
                                     '-s',
                                     '{}x{}'.format(self.PARAMS['size'], self.PARAMS['size']),
                                     '-vsync',
                                     '0',
                                     '-vframes',
                                     '100',
                                     output_format,
                                     '-hide_banner'
                                     ])


if __name__ == '__main__':
    obj = VideoFrameSampling()
    obj.process()
