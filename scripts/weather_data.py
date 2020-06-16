#Download the Weather data for specific location
#For instructions and API, check https://pypi.org/project/wwo-hist/

import numpy as np
from wwo_hist import retrieve_hist_data
from six.moves.urllib.parse import urlparse
#import urllib.request
from urlparse import urlparse

frequency=1
start_date = '1-Jan-2018'
end_date = '15-May-2020'
api_key = 'your key' #Change to the key you got from the API
location_list = ['Houston','Dallas','78205'] #Change to the location of interests

hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)
