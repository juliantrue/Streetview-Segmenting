import sys, os, time, subprocess
from optparse import OptionParser
import cv2

sys.path.append('./src')
from streetview import Core

parser = OptionParser() # Set up cmd line arg parsing
parser.add_option("-p", "--path", dest="raw_data_path",
                  help="Path to directory in which to save collected data.")
parser.add_option("-k", "--key", dest="API_KEY",
                  help="API_KEY for making calls to the Google streetview API")
parser.add_option("-u", "--url", dest="base_url",
                  default="https://maps.googleapis.com/maps/api/streetview?",
                  help="Path to directory in which to save collected data.")
parser.add_option("--logging_dir", dest="logging_dir",
                  help="Directory to store logs from Google API calls")
parser.add_option("--location", dest="location",
                  default="43.6576893,-79.3799391",
                  help="\"Latitude,Longitute\" of location to perform inferencing on.")

# Unpack CMD line options
(options, args) = parser.parse_args()

if not options.logging_dir:
    C = Core()
else:
    logging_dir = options.logging_dir
    C = Core(logging_dir)

raw_data_path = options.raw_data_path
base_url = options.base_url
API_KEY = options.API_KEY
location = (float(options.location.split(',')[0]),float(options.location.split(',')[1]))


# __main__
directory = C.get_by_location(base_url,API_KEY,location,save_to=raw_data_path)
inference_data_dir = directory

print("Images from {} stored in {}.".format(location,directory))

# Spawn subprocess instance of Mask_RCNN from Detectron via MakeFile
print("Spawning MASK RCNN instance...")
result = subprocess.run(["make","inference_mask_RCNN",
                         "--inference_dir \"{}\"".format(inference_data_dir)])
