import numpy as np
from wwo_hist import retrieve_hist_data
from six.moves.urllib.parse import urlparse
#import urllib.request
from urlparse import urlparse

frequency=1
start_date = '1-Jan-2018'
end_date = '15-May-2020'
api_key = 'ed92deaeb09b4a128d855844200505'
location_list = ['Houston','Dallas','78205']

hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)