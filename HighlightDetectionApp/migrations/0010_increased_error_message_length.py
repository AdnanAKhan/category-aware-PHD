# Generated by Django 2.1.7 on 2019-03-05 02:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('HighlightDetectionApp', '0009_remove_alt_title_from_metadata_table'),
    ]

    operations = [
        migrations.AlterField(
            model_name='youtubevideodownloadteststatus',
            name='message',
            field=models.CharField(default='', max_length=300),
        ),
        migrations.AlterField(
            model_name='youtubevideodownloadtrainstatus',
            name='message',
            field=models.CharField(default='', max_length=300),
        ),
    ]