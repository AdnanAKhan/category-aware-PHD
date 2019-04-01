from django.core.management.base import BaseCommand
from django.conf import settings
from HighlightDetectionApp.models import YoutubeVideoDownloadTestStatus, VideoMetaDataInfo, \
    YoutubeVideoDownloadTrainStatus
import logging
import os
import pafy as downloader
import json
import youtube_dl
import time

logger = logging.getLogger('app')


class Command(BaseCommand):
    help = 'Downloads the video'

    def add_arguments(self, parser):
        parser.add_argument('mode', type=str, help='Indicates which split we have to save video')

    def handle(self, *args, **kwargs):
        if kwargs['mode'] == 'train':
            logger.info('Loading training dataset')
            queryset = YoutubeVideoDownloadTrainStatus.objects.all()
        else:
            logger.info('Loading test dataset')
            queryset = YoutubeVideoDownloadTestStatus.objects.all()

        keywords_dest = os.path.join(settings.DATA_DIR, 'metadata', 'keywords')

        for obj in queryset.iterator():
            url = "https://www.youtube.com/watch?v={}".format(obj.youtube_id)
            try:
                dest_keyword = os.path.join(keywords_dest, '{}.json'.format(obj.youtube_id))
                if not os.path.exists(dest_keyword):
                    video = downloader.new(url)
                    try:
                        meta_data_obj = VideoMetaDataInfo()
                        meta_data_obj.youtube_id = obj.youtube_id
                        meta_data_obj.description = video.description if video.description else ''
                        meta_data_obj.title = video.title if video.title else ''
                        meta_data_obj.thumbnail_link = video.thumb if video.thumb else ''
                        meta_data_obj.category = video.category if video.category else ''
                        meta_data_obj.view_count = video.viewcount if video.viewcount else 0
                        meta_data_obj.dislike_count = video.dislikes if video.dislikes else 0
                        meta_data_obj.rating = video.rating if video.rating else 0
                        meta_data_obj.stream = ''

                        keywords = video.keywords if video.keywords else []

                        if not os.path.exists(dest_keyword):
                            with open(dest_keyword, 'w') as outfile:
                                json.dump(keywords, outfile)

                        meta_data_obj.save()
                    except Exception as ex:
                        logger.error('{}-{}'.format(obj.youtube_id, ex))
                else:
                    logger.info('{} exists'.format(obj.youtube_id))

            except OSError as ex:
                logger.error('{}-{}'.format(obj.youtube_id, ex))
