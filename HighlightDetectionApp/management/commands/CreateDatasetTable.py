from django.core.management.base import BaseCommand
from django.conf import settings
from HighlightDetectionApp.models import PersonalizedVideoTestDataset, PersonalizedVideoTrainingDataset
import logging
import os
import pandas as pd
import copy

logger = logging.getLogger('app')


class Command(BaseCommand):
    help = 'Populate  dataset table with training and testing data'

    def add_arguments(self, parser):
        parser.add_argument('mode', type=str, help='Indicates which dataset to create')

    def handle(self, *args, **kwargs):
        if kwargs['mode']:
            mode = kwargs['mode']
        else:
            mode = 'train'

        file_location = os.path.join(settings.DATA_DIR, 'personalized-highlights-dataset')

        if mode == 'train':
            MODEL = PersonalizedVideoTrainingDataset
            df = pd.read_csv(os.path.join(file_location, 'training.csv'), header=0)
        else:
            MODEL = PersonalizedVideoTestDataset
            df = pd.read_csv(os.path.join(file_location, 'testing.csv'), header=0)

        # clear all existing ones
        logger.debug("Deleting existing records")
        MODEL.objects.all().delete()

        list_of_rows = []

        for ind, row in df.iterrows():
            obj = MODEL()
            obj.youtube_id = row['youtubeId']
            obj.start = row['start']
            obj.duration = row['duration']
            obj.user_id = row['user_id']
            obj.video_duration = row['video_duration']
            obj.is_last = row['is_last']

            list_of_rows.append(copy.deepcopy(obj))

            if len(list_of_rows) % 2000 == 0:
                MODEL.objects.bulk_create(list_of_rows, batch_size=1000)
                list_of_rows = []
                logger.debug("{} dataset loaded - upto index {}".format(mode, ind))
            else:
                pass

        MODEL.objects.bulk_create(list_of_rows, batch_size=1000)
