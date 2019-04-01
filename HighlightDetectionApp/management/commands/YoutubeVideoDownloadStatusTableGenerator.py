from django.core.management.base import BaseCommand
from HighlightDetectionApp.models import PersonalizedVideoTestDataset, PersonalizedVideoTrainingDataset, \
    YoutubeVideoDownloadTestStatus, YoutubeVideoDownloadTrainStatus
import logging
import copy

logger = logging.getLogger('app')


class Command(BaseCommand):
    help = 'Populate  youtube video download  status table'

    def add_arguments(self, parser):
        parser.add_argument('mode', type=str, help='Indicates which dataset it is working on')

    def handle(self, *args, **kwargs):
        if kwargs['mode']:
            mode = kwargs['mode']
        else:
            mode = 'train'

        if mode == 'train':
            src_model = PersonalizedVideoTrainingDataset
            dest_model = YoutubeVideoDownloadTrainStatus

        else:
            src_model = PersonalizedVideoTestDataset
            dest_model = YoutubeVideoDownloadTestStatus

        unique_youtube_ids = src_model.objects.all().distinct('youtube_id')

        # clear all existing ones
        logger.debug("Deleting existing records")
        dest_model.objects.all().delete()

        list_of_youtube_objects = []
        count = 0
        for src_obj in unique_youtube_ids.iterator():
            obj = dest_model()
            obj.youtube_id = src_obj.youtube_id
            obj.start = False

            list_of_youtube_objects.append(copy.deepcopy(obj))

            if len(list_of_youtube_objects) % 2000 == 0:
                count += 1
                dest_model.objects.bulk_create(list_of_youtube_objects, batch_size=1000)
                list_of_youtube_objects = []
                logger.debug("{}: batch of {} done".format(mode, count))
            else:
                pass

        dest_model.objects.bulk_create(list_of_youtube_objects, batch_size=1000)
