from django.db import models


# Create your models here.


class PersonalizedVideoDatasetAbstract(models.Model):
    """
    Personalized Dataset model
    """
    youtube_id = models.CharField(max_length=20, db_index=True)
    start = models.FloatField(max_length=10)
    duration = models.FloatField(max_length=10)
    user_id = models.CharField(max_length=15)
    video_duration = models.FloatField(max_length=20)
    is_last = models.BooleanField(default=False)

    class Meta:
        abstract = True

    def __str__(self):
        return "{} - {}".format(self.youtube_id, self.user_id)


class PersonalizedVideoTrainingDataset(PersonalizedVideoDatasetAbstract):
    class Meta:
        db_table = u'PersonalizedVideoTrainingDataset'


class PersonalizedVideoTestDataset(PersonalizedVideoDatasetAbstract):
    class Meta:
        db_table = u'PersonalizedVideoTestDataset'


class YoutubeVideoDownloadStatusAbstract(models.Model):
    youtube_id = models.CharField(max_length=20, unique=True, db_index=True)
    downloaded = models.BooleanField(default=False)
    message = models.CharField(max_length=300, default='')

    class Meta:
        abstract = True

    def __str__(self):
        return "{} - {}".format(self.youtube_id, self.downloaded)


class YoutubeVideoDownloadTrainStatus(YoutubeVideoDownloadStatusAbstract):
    class Meta:
        db_table = u'YoutubeVideoDownloadTrainStatus'


class YoutubeVideoDownloadTestStatus(YoutubeVideoDownloadStatusAbstract):
    class Meta:
        db_table = u'YoutubeVideoDownloadTestStatus'


class VideoMetaDataInfo(models.Model):
    youtube_id = models.CharField(max_length=20, unique=True, db_index=True)
    description = models.CharField(max_length=8000, default='')
    title = models.CharField(max_length=1000, default='')
    thumbnail_link = models.CharField(max_length=100, default='')
    category = models.CharField(max_length=100, null=True)
    view_count = models.IntegerField(default=0)
    like_count = models.IntegerField(default=0)
    dislike_count = models.IntegerField(default=0)
    rating = models.FloatField(default=0.0)
    stream = models.CharField(max_length=20, default='')

    class Meta:
        db_table = u'VideoMetadataInformation'

    def __str__(self):
        return '{} - {}'.format(self.youtube_id, self.title)
