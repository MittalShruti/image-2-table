import json

# from dataclasses import dataclass
# from functools import cached_property
# import typing
# import cv2
# import numpy as np
# import io
# from pathlib import Path
# from typing import Tuple, Union, Dict, List, Optional, OrderedDict, NamedTuple, Any, Callable, Set
# import math
# import copy
# from collections import OrderedDict
# from dataclasses import dataclass, field
# from collections import defaultdict
# from itertools import groupby
# from operator import itemgetter
# import itertools
# from functools import partial
# import random
# from queue import PriorityQueue
# import polars as pl
# import fitz

# from wow import Image
import boto3
# from io import BytesIO

import img2table


def lambda_handler(event, context):

    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='bankstatement-pdf2', Key='b.jpg')
    image_bytes = response['Body'].read()

    img = Image(image_bytes)
    tables = img.extract_tables(borderless_tables=True)
    print(tables)

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
