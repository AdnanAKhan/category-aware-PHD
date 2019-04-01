from django.core.management.base import BaseCommand
from django.conf import settings
from HighlightDetectionApp.models import YoutubeVideoDownloadTestStatus, VideoMetaDataInfo
import logging
import os
import pafy as downloader
import json
import youtube_dl

logger = logging.getLogger('app')


class Command(BaseCommand):
    help = 'Downloads the video'

    def add_arguments(self, parser):
        parser.add_argument('youtube_id', type=str, help='Indicates which split we have to save video')

    def handle(self, *args, **kwargs):
        if kwargs['youtube_id']:
            youtube_id = kwargs['youtube_id']
        else:
            youtube_id = 'youtube_id'

        keywords_dest = os.path.join(settings.DATA_DIR, 'metadata', 'keywords')
        video_dest = os.path.join(settings.DATA_DIR, 'video')

        url = "https://www.youtube.com/watch?v={}".format(youtube_id)
        logger.debug('Working on {}'.format(youtube_id))
        try:
            video = downloader.new(url)
            download = False

            # select the stream
            selected_stream = None
            for stream in video.videostreams:
                if stream.extension == 'mp4':
                    selected_stream = stream
                    break

            if selected_stream:
                try:
                    dest_video = os.path.join(video_dest, '{}.mp4'.format(youtube_id))

                    if not os.path.exists(dest_video):
                        selected_stream.download(filepath=dest_video)

                    meta_data_obj = VideoMetaDataInfo()
                    meta_data_obj.youtube_id = youtube_id
                    meta_data_obj.description = video.description
                    meta_data_obj.title = video.title
                    meta_data_obj.thumbnail_link = video.thumb
                    meta_data_obj.category = video.category
                    meta_data_obj.view_count = video.viewcount
                    meta_data_obj.dislike_count = video.dislikes
                    meta_data_obj.rating = video.rating
                    meta_data_obj.stream = selected_stream.resolution

                    keywords = video.keywords
                    dest_keyword = os.path.join(keywords_dest, '{}.json'.format(youtube_id))
                    if not os.path.exists(dest_keyword):
                        with open(dest_keyword, 'w') as outfile:
                            json.dump(video.keywords, outfile)

                    download = True
                except Exception as ex:
                    logger.error(ex)
            else:
                message = 'mp4 format is not available'
                logger.error(message)

            if download:
                downloaded = True
                message = 'success'
                logger.error(message)

            logger.debug('completed on {}'.format(youtube_id))

        except OSError as ex:
            logger.info('{}'.format(ex))
