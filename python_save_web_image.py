# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:02:24 2018

@author: Administrator
"""

import urllib
import urllib.request
import re

def download_page(url):
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    data = response.read()
    return data

url = 'https://www.hao123.com/?tn=93620501_hao_pg'
html = download_page(url)

regx = r'src="(.*?.(jpg|png))"'
pattern = re.compile(regx)

get_image = re.findall(pattern,repr(html))
http = 'https'
x=132
for imgurl in get_image:
#    f = open(str(x)+'.jpg','w')
    if http != imgurl[0][:5]:
        continue
    imgurl = imgurl[0]
    data = download_page(imgurl)
    urllib.request.urlretrieve(imgurl,r'D:\xx\%s.jpg' %x)
    x+=1


from PIL import Image
im_path = r'D:\CNY_IMAGE\脏污\0131114839_15\F_R_F_Correct.bmp'
im = Image.open(im_path)
im.resize((100,40))
