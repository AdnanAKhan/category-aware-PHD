from django.core.management.base import BaseCommand
from django.conf import settings
from HighlightDetectionApp.models import YoutubeVideoDownloadTestStatus, VideoMetaDataInfo
import logging
import os
import subprocess

logger = logging.getLogger('app')


class Command(BaseCommand):
    help = 'Downloads the video'

    def add_arguments(self, parser):
        parser.add_argument('mode', type=str, help='Indicates which split we have to save video')

    def handle(self, *args, **kwargs):
        if kwargs['mode']:
            mode = kwargs['mode']
        else:
            mode = 'train'

        video_dest = os.path.join(settings.DATA_DIR, 'youtube_dl')

        queryset = YoutubeVideoDownloadTestStatus.objects.filter(downloaded=False).exclude(message__in=['OSError'])

        for obj in queryset.iterator():
            url = "https://www.youtube.com/watch?v={}".format(obj.youtube_id)

            download = False
            try:
                p = subprocess.Popen(
                    "youtube-dl  -f '(mp4)[filesize<100M]' -r 10M -o {}/'%(id)s.%(ext)s' {}".format(video_dest, url),
                    stdout=subprocess.PIPE, shell=True)
                (output, err) = p.communicate()
                process_status = p.wait()

                if process_status == 0:
                    download = True
                else:
                    logger.error('{}-{}'.format(obj.youtube_id, err))

            except Exception as ex:
                logger.error('{}-{}'.format(obj.youtube_id, ex))

            if download:
                obj.downloaded = True
                obj.message = 'success'
                obj.save()
                logger.debug('{} downloaded successfully'.format(obj.youtube_id))

