import os, shutil, logging, math
from collections import OrderedDict
import requests
import cv2
from .logging_facility import LoggingWrapper


"""
Usage:
location1: type tuple: (lat1, lon1)
location2: type tuple: (lat2, lon2)
Based on Haversine formula found here:
https://en.wikipedia.org/wiki/Haversine_formula
returns: result: type float: distance in meters
"""
def delta_lat_lon_to_meters(location1, location2):
    E_radius = 6378.137 # ~Earth's radius in kilometers
    d_lat = (location2[0]*math.pi/180) - (location1[0]*math.pi/180)
    d_lon = (location2[1]*math.pi/180) - (location1[1]*math.pi/180)

    a =  math.sin(d_lat/2)*math.sin(d_lat/2) + \
    math.cos(location1[0]*math.pi / 180)*math.cos(location2[0]*math.pi / 180) * \
    math.sin(d_lon/2)*math.sin(d_lon/2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R*c
    return d * 1000

"""
Usage:
curr_location: type tuple (lat, lon
dx: type float: change in x in meters
dy: type float: change in y in meters

returns: type tuple: location(lat, lon)
"""
def meters_to_lat_lon(curr_location, dx, dy):
    E_radius = 6378.137 # ~Earth's radius in kilometers
    delta_lat = curr_location[0] + (dy / E_radius) * (180 / math.pi)
    delta_lon = curr_location[1] + (dx / E_radius) * (180 / math.pi) / \
                cos(curr_location[0] * math.pi/180)

    return (new_lat, new_lon)

"""
Saving helper function for streamed data from requests
"""
def stream_save(r, directory, save_to_file):
    try:
        os.mkdir(directory)
    except FileExistsError as e:
        pass
    with open('{}.png'.format(save_to_file), 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)

"""
Core functionality of the module
"""
class Core(object):
    def __init__(self,logs_folder=None):
        self.L = LoggingWrapper(log_folder_path=logs_folder)
        self.logger = logging.getLogger('Streetview_Module')
        self.logger.info("Streetview Module Initialized")

    """
    Usage:
    Pass in the base url on which to build the request on, followed by the API_KEY
    and the signature if needed. The request builder then takes as many kwargs as
    needed.
    Returns Request string
    """
    def request_builder(self, BASE_URL, API_KEY, kwargs, signature=None):
        request = BASE_URL
        for key in kwargs:
            request += "{}={}&".format(key,kwargs[key])

        request += "key={}".format(API_KEY)
        if(not(signature is None)):
            request += "&signature={}".format(signature)
        return request

    """
    Usage:
    See request builder. Builds request for metadata.
    Run this prior to sending image request to google servers. Confirms image
    availability as well as request validation.
    """
    def metadata_request_builder(self, BASE_URL, API_KEY, kwargs, signature=None):
        request = BASE_URL
        request = request[:-1] + "/" + "metadata?"
        for key in kwargs:
            request += "{}={}&".format(key,kwargs[key])

        request += "key={}".format(API_KEY)
        if(not(signature is None)):
            request += "&signature={}".format(signature)
        return request

    """
    Usage:
    Requires tuple of geographic coordinates in the format (lon,lat)
    Returns:
    List of images associated with that location unless save_to parameter is
    defined

    Example
    location = (43.656009, -79.380354)
    """
    def get_by_location(self, BASE_URL, API_KEY, location, save_to=None,
                        size=(600,400), outdoor_only=True, signature=None):
        if(not(type(location) is tuple)):
            raise Exception("\'location\' must be of type tuple.")
        if(not(type(size) is tuple)):
            raise Exception("\'size\' must be of type tuple.")

        # Remove brackets from tuple input and convert to strings
        size_s = str(size[0])+"x"+str(size[1])
        loc_s = str(location)[1:][:-1]
        headings = [0, 90, 180, 270] # N E W S
        source_s = "outdoor" if outdoor_only else "default"

        # Memory for images
        imgs = []

        user_repsonse = input("Are you sure you want to download {} images?(yes/no): ".format(len(headings)))
        if(not(user_repsonse == "yes")):
            raise Exception("User did not confirm image download.")
        directory = "" # Placeholder for returned path to data
        # Build kwargs in order
        for heading in headings:
            head_s = str(heading)
            kwargs = OrderedDict([('size', size_s), ('location', loc_s),
                                  ('heading', head_s), ('source', source_s)])
            self.logger.info("Kwargs: {}".format(kwargs))

            # Request image metadata
            meta_req = self.metadata_request_builder(BASE_URL, API_KEY, kwargs)
            self.logger.info("Sending image metadata request: {}".format(meta_req))
            meta_r = requests.get(meta_req)
            response = meta_r.json()

            if self.L.debug_mode:
                self.logger.debug("Response: {}".format(meta_r.text))
            if(not(str(response['status']) == "OK")):
                # Make noise if response is not OK
                raise Exception("Request status: {}".format(response['status']))

            # Request for each cardinal direction heading
            req = self.request_builder(BASE_URL, API_KEY, kwargs)
            to_file = req.split("&")[1]+req.split("&")[2]

            # Check if file already exists
            exists = False
            if not save_to == None:
                directory = os.path.join(save_to,req.split("&")[1])
                save_to_file = os.path.join(directory,to_file)
                exists = os.path.isfile(save_to_file + ".png")

                # If the file doesn't already exists, GET from API
                if not exists:
                    self.logger.info("Sending image request: {}".format(req))
                    r = requests.get(req, stream=True)

                    # Save to file
                    stream_save(r,directory,save_to_file)
                    del r

            # Save to temp file then to opencv img obj
            else:
                self.logger.info("Sending image request: {}".format(req))
                r = requests.get(req, stream=True)
                with open('./temp.png', 'wb') as out_file:
                    shutil.copyfileobj(r.raw, out_file)
                del r
                img = cv2.imread('./temp.png')
                imgs.append(img)
                os.remove('./temp.png')

        if save_to == None:
            return imgs
        else:
            return directory

    """
    Usage:
    Requires address in string format. Address may resemble a google maps query
    or just the actual address.
    Returns:
    List of images associated with that address
    Optional: return only the first n images by specifying n.

    Example
    search_string = "245 Church St, Toronto, ON M5B 2K3"
    imgs = get_by_search(search_string, n=4)
    """
    def get_by_search(self, BASE_URL, API_KEY, search_string, save_to=None,
                      size=(600,400), outdoor_only=True, signature=None):
        if(not(type(search_string) is type("string"))):
            raise Exception("\'location\' must be of type string.")
        if(not(type(size) is tuple)):
            raise Exception("\'size\' must be of type tuple.")

        # Convert to strings
        size_s = str(size[0])+"x"+str(size[1])
        loc_s = search_string.replace(" ", "%20")
        headings = [0, 90, 180, 270] # N E W S
        source_s = "outdoor" if outdoor_only else "default"


        # Memory for images
        imgs = []

        user_repsonse = input("Are you sure you want to download {} images?(yes/no): ".format(len(headings)))
        if(not(user_repsonse == "yes")):
            raise Exception("User did not confirm image download.")
        directory = "" # Placeholder for returned path to data
        # Build kwargs in order
        for heading in headings:
            head_s = str(heading)
            kwargs = OrderedDict([('size', size_s), ('location', loc_s),
                                  ('heading', head_s), ('source', source_s)])

            # Request image metadata
            meta_req = self.metadata_request_builder(BASE_URL, API_KEY, kwargs)
            self.logger.info("Sending image metadata request: {}".format(meta_req))
            meta_r = requests.get(meta_req)
            response = meta_r.json()
            if self.L.debug_mode:
                self.logger.debug("Response: {}".format(meta_r.text))
            if(not(str(response['status']) == "OK")):
                raise Exception("Request status: {}".format(response['status']))

            # Request for each cardinal direction heading
            req = self.request_builder(BASE_URL, API_KEY, kwargs)
            to_file = req.split("&")[1]+ req.split("&")[2]

            # Check if file already exists
            exists = False
            if not save_to == None:
                directory = os.path.join(save_to,req.split("&")[1])
                save_to_file = os.path.join(directory,to_file)
                exists = os.path.isfile(save_to_file + ".png")

                # If the file doesn't already exists, GET from API
                if not exists:
                    self.logger.info("Sending image request: {}".format(req))
                    r = requests.get(req, stream=True)

                    # Save to file
                    stream_save(r,directory,save_to_file)
                    del r

            # Save to temp file then to opencv img obj
            else:
                self.logger.info("Sending image request: {}".format(req))
                r = requests.get(req, stream=True)
                with open('./temp.png', 'wb') as out_file:
                    shutil.copyfileobj(r.raw, out_file)
                del r
                img = cv2.imread('./temp.png')
                imgs.append(img)
                os.remove('./temp.png')

        if save_to == None:
            return imgs
        else:
            return directory

    """
    Usage:
    Requires:
    base_url
    API_KEY
    location: tuple of geographic coordinates in the format (lon,lat)
    radius: radius in metres around location center to get images from
    Returns:
    ALL available image in the given radius

    Example
    location = (43.656009, -79.380354), radius = 10
    """
    def get_all_in_area(self, BASE_URL, API_KEY, location, radius, save_to=None,
                        size=(600,400), outdoor_only=True, signature=None):
        pass
