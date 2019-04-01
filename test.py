#!/usr/bin/python
## get subprocess module 
import subprocess

## call date command ##
p = subprocess.Popen("youtube-dl  -f '(mp4)[filesize<100M]' -o '%(id)s.%(ext)s' https://www.youtube.com/watch?v=6qqvVVuPbdE", stdout=subprocess.PIPE, shell=True)
(output, err) = p.communicate()

## Wait for date to terminate. Get return returncode ##
p_status = p.wait()
print("Command output : ", output)
print("Command exit status/return code : ", p_status)