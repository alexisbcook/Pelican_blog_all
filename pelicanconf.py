#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Alexis Cook'
SITENAME = 'Alexis Cook'
GITHUB_URL = 'https://github.com/alexisbcook'
GOOGLE_ANALYTICS_CODE = 'UA-83225772-1'

PATH = 'content'
TIMEZONE = 'America/Mexico_City'
DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = None
SOCIAL = (('github', 'https://github.com/alexisbcook'),
		  ('linkedin', 'https://www.linkedin.com/in/alexis-cook-a6127753'),
          ('twitter', 'https://twitter.com/alexis_b_cook'),)

# Theme is pelican-bootstrap3
#THEME = 'pelican-themes/pelican-bootstrap3'
THEME = 'theme'
PYGMENTS_STYLE = 'trac'
# like trac > borland > zenburn
BOOTSTRAP_THEME = 'yeti'
# like yeti > cosmo > lumen > flatly

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

# for ipython notebooks
MARKUP = ('md', 'ipynb')
PLUGIN_PATHS = './plugins'
PLUGINS = ['ipynb.markup', 'pelican_fontawesome']

# format individual articles
SHOW_ARTICLE_AUTHOR = False
SHOW_ARTICLE_CATEGORY = False
SHOW_DATE_MODIFIED = False
DISPLAY_ARTICLE_INFO_ON_INDEX = False


# format top menu bar
DISPLAY_PAGES_ON_MENU = True
DISPLAY_CATEGORIES_ON_MENU = False



# format sidebar
AVATAR = '/images/avatar.jpg'
HIDE_SIDEBAR = False
DISPLAY_TAGS_ON_SIDEBAR = False
DISPLAY_CATEGORIES_ON_SIDEBAR = False
BOOTSTRAP_NAVBAR_INVERSE = True
DISPLAY_RECENT_POSTS_ON_SIDEBAR = True


# format whole website
BOOTSTRAP_FLUID = False


# images and pdfs exported to output
STATIC_PATHS = ['images', 'pdfs']


# daniel rodriguez theme
PAGE_DIRS = ['pages']
ARTICLE_DIRS = ['articles']
ABOUT_PAGE = '/pages/about-me.html'
#GITHUB_USERNAME = 'alexisbcook'
#TWITTER_USERNAME = 'alexis_b_cook'
SHOW_ARCHIVES = True
#PAGE_SAVE_AS = '{category}/{slug}.html'
#PAGE_URL = PAGE_SAVE_AS
IPYNB_EXTEND_STOP_SUMMARY_TAGS = [('h2', None), ('ol', None), ('ul', None)]
MD_EXTENSIONS = ['codehilite(css_class=codehilite)', 'extra']
DEFAULT_DATE_FORMAT = '%B %d, %Y'
SUMMARY_MAX_LENGTH = 150
DEFAULT_PAGINATION = 10

NAVIGATION = [
    # You probably want to fill these in so they point to your user pages
    {'site': 'github', 'user': 'alexisbcook', 'url': 'https://github.com/alexisbcook'},
    {'site': 'twitter', 'user': 'alexis_b_cook', 'url': 'https://twitter.com/alexis_b_cook'},
    {'site': 'linkedin', 'user': 'alexis-cook', 'url': 'https://www.linkedin.com/in/alexis-cook-a6127753'},
    {'site': 'instagram', 'user': 'alexis.b.cook', 'url': 'https://www.instagram.com/alexis.b.cook'},
    {'site': 'angellist', 'user': 'alexis_cook', 'url': 'https://angel.co/alexis_cook'},
]

# Disqus
DISQUS_SITENAME = "alexisbcook-github-io"
DISQUSURL = 'https://alexisbcook.github.io/'