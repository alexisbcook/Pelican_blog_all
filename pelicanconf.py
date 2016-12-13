#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Alexis Cook'
SITENAME = 'Alexis Cook'
GITHUB_URL = 'https://github.com/alexisbcook'
GOOGLE_ANALYTICS_ID = 'UA-83225772-1'

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
THEME = 'pelican-themes/pelican-bootstrap3'
PYGMENTS_STYLE = 'trac'
# like trac > borland > zenburn
BOOTSTRAP_THEME = 'yeti'
# like yeti > cosmo > lumen > flatly


# ipython notebook support
PLUGIN_PATHS = ['./plugins']
MARKUP = ('md')
PLUGINS = ['liquid_tags.notebook']
NOTEBOOK_DIR = 'notebooks'
EXTRA_HEADER = open('_nb_header.html').read().decode('utf-8')


# format individual articles
SHOW_ARTICLE_AUTHOR = False
SHOW_ARTICLE_CATEGORY = False
SHOW_DATE_MODIFIED = False
DISPLAY_ARTICLE_INFO_ON_INDEX = False


# to make this file work naahhhssseee
STATIC_PATHS = ['images', 'pdfs']
RELATIVE_URLS = True



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


# Disqus
DISQUS_SITENAME = "alexisbcook-github-io"
DISQUSURL = 'https://alexisbcook.github.io/'



# currently not using
#TAG_CLOUD_MAX_ITEMS = 10
#DISPLAY_TAGS_INLINE = True
#GITHUB_USER = 'alexisbcook'
#GITHUB_SKIP_FORK = True
#GITHUB_REPO_COUNT = 5
#GITHUB_SHOW_USER_LINK = False
#ABOUT_ME = '''<strong>Alexis Cook</strong>
#	<br />alexis.cook@gmail.com
#	<br />
#	<br />Explorer, Learner, Builder
#	'''

