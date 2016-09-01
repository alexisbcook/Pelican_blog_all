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

# Social widget
SOCIAL = (('github', 'https://github.com/alexisbcook'),
		  ('linkedin', 'https://www.linkedin.com/in/alexis-cook-a6127753'),
          ('twitter', 'https://twitter.com/alexis_nunez_b'),)

RELATIVE_URLS = True

THEME = 'pelican-themes/pelican-bootstrap3'

TAG_CLOUD_MAX_ITEMS = 10
DISPLAY_TAGS_INLINE = True

SHOW_ARTICLE_AUTHOR = False
SHOW_ARTICLE_CATEGORY = False
SHOW_DATE_MODIFIED = False

PYGMENTS_STYLE = 'trac'
# like trac > borland > zenburn
BOOTSTRAP_THEME = 'yeti'
# like yeti > cosmo > lumen > flatly

DISPLAY_PAGES_ON_MENU = True
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_TAGS_ON_SIDEBAR = True
DISPLAY_CATEGORIES_ON_SIDEBAR = True
BOOTSTRAP_NAVBAR_INVERSE = True
DISPLAY_ARTICLE_INFO_ON_INDEX = False

#GITHUB_USER = 'alexisbcook'
#GITHUB_SKIP_FORK = True
#GITHUB_REPO_COUNT = 5
#GITHUB_SHOW_USER_LINK = False

#BANNER = '/content/images/banner.png'
#BANNER_SUBTITLE = 'not sure what this does either'


NOTEBOOK_DIR = 'content/notebooks'
PLUGIN_PATHS = ['./plugins']
#PLUGINS = ['tag_cloud', 'liquid_tags.img', 'liquid_tags.video',
#			'liquid_tags.youtube', 'liquid_tags.include_code',
#           	'liquid_tags.notebook','render_math']
MARKUP = ('md', 'ipynb')
PLUGINS = ['tag_cloud', 'ipynb.markup']
HIDE_SIDEBAR = True

BOOTSTRAP_FLUID = False
DISQUS_SITENAME = "alexisbcook-github-io"
DISQUSURL = 'https://alexisbcook.github.io/'
