from __future__ import unicode_literals
import youtube_dl

fname = "links.txt"

ydl_opts = {
            'format': 'bestaudio/best',
                'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                                    'preferredcodec': 'wav',
                                            'preferredquality': '192',
                                                }],
                }
count=1
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    with open(fname, 'r') as f:
        for line in f:
            ydl.download([line])
            print(count)
            count+=1
