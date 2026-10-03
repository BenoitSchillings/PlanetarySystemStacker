# -*- coding: utf-8; -*-
"""
Copyright (c) 2018 Rolf Hempel, rolf6419@gmx.de

This file is part of the PlanetarySystemStacker tool (PSS).
https://github.com/Rolf-Hempel/PlanetarySystemStacker

PSS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSS.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
import mmap
import cv2
import struct
import datetime
import numpy as np


class SERParser(object):

    __author__ = 'Michal Powalko'
    __author_email__ = 'm.powalko@gmail.com'
    __version__ = '1.1'
    __name__ = 'SER parser for PlanetarySystemStacker tool (PSS)'

    # Size of the fixed SER file header in bytes.
    HEADER_SIZE = 178

    def __init__(self, ser_file, SER_16bit_shift_correction=True):
        """
        Parse video files of type SER (8 or 16 bit). Provide access to individual frames based on
        the frame index.

        :param ser_file: Full name of the video file.
        :param SER_16bit_shift_correction: If True and the frame type is 16bit, the video frames
                                           are analyzed to find the number of unused high bits in
                                           pixel data. In read operations data are shifted up by
                                           this number of bits.
        """
        super().__init__()

        self.warn_message = self.sanity_check(ser_file)

        self.fid = self.open_file(ser_file)

        self.header = self.read_header()

        self.frame_size = self.header['ImageWidth'] * \
                          self.header['ImageHeight'] * \
                          self.header['BytesPerPixel']

        # Truncated captures (e.g. recording aborted, disk full) contain fewer frames than stated
        # in the header. Clamp the frame count to the number of complete frames in the file.
        file_size = os.fstat(self.fid.fileno()).st_size
        self.frame_count = self.header['FrameCount']
        if self.frame_size > 0:
            self.frame_count = max(0, min(self.frame_count,
                                          (file_size - self.HEADER_SIZE) // self.frame_size))
        else:
            self.frame_count = 0

        # This parameter is the number of unused bits in pixel values. It will be determined
        # experimentally later, based on a sample of frames. This is necessary because SER headers
        # often contain incorrect values for PixelDepthPerPlane. Often this value is set to 16, but
        # the image data only contain 8, 10 or 12 bits.
        self.shift_pixels = 0
        self.shift_limit = 65535
        self.shift_factor = (1., 1., 1., 1.)

        # If True, 16bit pixel data are big-endian, and bytes are swapped in read operations.
        self.byteswap = False

        if self.header['PixelDepthPerPlane'] <= 8:
            self.PixelDepthPerPlane = np.dtype(np.uint8)
        else:
            # FireCapture uses "LittleEndian".
            # Until FireCatpure 2.7 this flag was not set properly.
            self.PixelDepthPerPlane = np.dtype(np.uint16).newbyteorder('<')

        # Map the frame data area of the file into memory. Frames are returned as (read-only)
        # views into this map, which avoids a seek / read / copy per frame.
        self.mmap = None
        self.frame_data = None
        self.map_frame_data()

        if self.header['PixelDepthPerPlane'] > 8:
            # The "LittleEndian" header flag cannot be trusted: The SER specification (v3) says
            # 1 = little-endian, but most capture programs (including the original Lucam recorder)
            # write 0 for little-endian data. Therefore the flag is ignored and little-endian is
            # kept as the default. Only if the image data strongly look big-endian, bytes are
            # swapped (see "detect_big_endian").
            # The (native) dtype is kept, the bytes of each frame are swapped when read.
            self.byteswap = self.detect_big_endian()

            # Test how many of the 16 bits are not used. Set the parameter which is used from now
            # on to shift pixel values such that the full 16bit range is used.
            if SER_16bit_shift_correction:
                self.correct_dynamic_range()

        self.color = 8 <= self.header['ColorID'] <= 19 and self.header['DebayerPattern'] is not None \
                     or 100 <= self.header['ColorID'] <= 101

    def sanity_check(self, ser_file):
        warn_message = None
        if not os.path.isfile(ser_file):
            raise IOError("File does not exist")
        elif os.stat(ser_file).st_size == 0:
            raise IOError("File is empty")
        else:
            with open(ser_file, 'rb') as fid:
                HEADER = fid.read(14).decode()
            if HEADER != 'LUCAM-RECORDER':
                warn_message = "File does not conform to SER format, first 14 characters of header are: '" + HEADER + "'"
        return warn_message

    def open_file(self, ser_file):
        return open(ser_file, 'rb')

    def map_frame_data(self):
        """
        Create a memory map of the file and a numpy array view "frame_data" of the frame data area,
        with shape (frame_count, height, width) or (frame_count, height, width, planes).

        :return: -
        """

        if self.frame_count == 0:
            return
        self.mmap = mmap.mmap(self.fid.fileno(), 0, access=mmap.ACCESS_READ)
        if self.header['NumberOfPlanes'] == 1:
            shape = (self.frame_count, self.header['ImageHeight'], self.header['ImageWidth'])
        else:
            shape = (self.frame_count, self.header['ImageHeight'], self.header['ImageWidth'],
                     self.header['NumberOfPlanes'])
        self.frame_data = np.frombuffer(self.mmap, dtype=self.PixelDepthPerPlane,
                                        count=int(np.prod(shape)),
                                        offset=self.HEADER_SIZE).reshape(shape)

    def detect_big_endian(self):
        """
        Heuristic test whether 16bit pixel data are stored in big-endian byte order. Natural images
        are smooth: Neighbouring pixels (of the same Bayer color, hence a stride of two) differ by
        much less than the spread of pixel values. If the byte order is wrong, the noisy low byte
        ends up in the high byte, and neighbouring values become essentially uncorrelated. Compare
        this smoothness measure for both byte orders and only report big-endian if the swapped
        version is much smoother. Note that for data using only one byte (8bit values in a 16bit
        container) both measures are identical, and little-endian is kept.

        :return: True, if the data are very likely big-endian; False otherwise.
        """

        if self.frame_data is None:
            return False

        def roughness(frame):
            frame = frame.astype(np.float32)
            std = frame.std()
            if std == 0.:
                return None
            return np.abs(frame[:, 2:] - frame[:, :-2]).mean() / std

        # Use a subsample of rows of the middle frame. Frames are 2D or 3D (planes last).
        frame = self.frame_data[self.frame_count // 2][::4]
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        if frame.shape[1] < 3:
            return False
        r_native = roughness(frame)
        r_swapped = roughness(frame.byteswap())
        if r_native is None or r_swapped is None:
            return False
        return r_swapped < 0.25 * r_native

    def read_header(self):
        """
        Read the "Header" of SER file with fixed size of 178 Byte.

        :return:    header              Dictionary containing the both "raw"
                                        and "decoded" values of "header"
        """

        KEYS = ('FileId', 'LuID', 'ColorID', 'LittleEndian', 'ImageWidth',
                'ImageHeight', 'PixelDepthPerPlane', 'FrameCount', 'Observer',
                'Instrument', 'Telescope', 'DateTime', 'DateTime_UTC')

        ColorID = {0:   'Grayscale',
                   8:   'Force Bayer RGGB',
                   9:   'Force Bayer GRBG',
                   10:  'Force Bayer GBRG',
                   11:  'Force Bayer BGGR',
                   16:  'BAYER_CYYM',
                   17:  'BAYER_YCMY',
                   18:  'BAYER_YMCY',
                   19:  'BAYER_MYYC',
                   100: 'RGB',
                   101: 'BGR'}

        header = {key: value.decode('latin1') if isinstance(value, bytes) else
                  value for (key, value) in zip(KEYS, struct.unpack(
                          '<14s 7i 40s 40s 40s 2q', self.fid.read(self.HEADER_SIZE)))}

        if header['ColorID'] == 8:
            header['DebayerPattern'] = cv2.COLOR_BayerRG2BGR
        elif header['ColorID'] == 9:
            header['DebayerPattern'] = cv2.COLOR_BayerGR2BGR
        elif header['ColorID'] == 10:
            header['DebayerPattern'] = cv2.COLOR_BayerGB2BGR
        elif header['ColorID'] == 11:
            header['DebayerPattern'] = cv2.COLOR_BayerBG2BGR
        else:
            header['DebayerPattern'] = None

        header['ColorIDDecoded'] = ColorID.get(header['ColorID'], 'Unknown')

        if header['ColorID'] < 100:
            header['NumberOfPlanes'] = 1
        else:
            header['NumberOfPlanes'] = 3

        if header['PixelDepthPerPlane'] <= 8:
            header['BytesPerPixel'] = header['NumberOfPlanes']
            if header['ColorID'] < 100:
                header['PixelDataOrganization'] = 'M'
            elif header['ColorID'] == 100:
                header['PixelDataOrganization'] = 'R G B'
            else:
                header['PixelDataOrganization'] = 'B G R'
        else:
            header['BytesPerPixel'] = 2 * header['NumberOfPlanes']
            if header['ColorID'] < 100:
                header['PixelDataOrganization'] = 'MM'
            elif header['ColorID'] == 100:
                header['PixelDataOrganization'] = 'RR GG BB'
            else:
                header['PixelDataOrganization'] = 'BB GG RR'

        try:
            header['DateTime_Decoded'] = datetime.datetime(1, 1, 1) + \
                datetime.timedelta(microseconds=header['DateTime'] // 10)
        except Exception:
            header['DateTime_Decoded'] = None

        try:
            header['DateTime_UTC_Decoded'] = datetime.datetime(1, 1, 1) + \
                datetime.timedelta(microseconds=header['DateTime_UTC'] // 10)
        except Exception:
            header['DateTime_UTC_Decoded'] = None

        # Check, if FireCature metadata is available. Malformed metadata are ignored, they must
        # not prevent the file from being opened.
        if 'fps=' in header['Telescope']:
            try:
                header['FPS'] = float(header['Telescope'].split('fps=')[1].split('gain')[0])
                header['Gain'] = int(header['Telescope'].split('gain=')[1].split('exp')[0])
                header['Exposure [ms]'] = float(
                    header['Telescope'].split('exp=')[1].split('\x00')[0])
            except Exception:
                pass

        self.frame_number = -1

        return header

    def frame_view(self, frame_number=None):
        """
        Check the frame index and return a (read-only) view of the raw frame data in the memory map.

        :param frame_number: Frame index (optional). If None, the next frame is returned.
        :return: Numpy array view of the frame data (2D for one plane, 3D otherwise).
        """

        if frame_number is None:
            frame_number = self.frame_number + 1

        if not 0 <= frame_number < self.frame_count:
            raise IOError('Error in reading SER frame, index: {0} is out of bounds'.format(frame_number))

        self.frame_number = frame_number

        if self.byteswap:
            return self.frame_data[frame_number].byteswap()
        return self.frame_data[frame_number]

    def read_frame_raw(self, frame_number=None):
        """
        Read the "Image Data" of SER file. Return the 2D or 3D image data without changing the
        content (e.g. debayering or conversion to / from grayscale).

        :return:    image_data: Multi dimmensional Numpy array containig image frame data.
        """

        frame = self.frame_view(frame_number)

        # If the pixel values do not use the full dynamic range, shift them accordingly.
        if self.shift_pixels:
            # The shift has been estimated from a sample of frames. If a pixel would overflow,
            # saturate it at 65535 instead of wrapping around (i.e. multiply with saturation).
            if frame.max() > self.shift_limit:
                return cv2.multiply(frame, self.shift_factor)
            return frame << self.shift_pixels
        else:
            return frame

    def read_frame(self, frame_number=None):
        """
        Read the "Image Data" of SER file.

        :return:    image_data: Multi dimmensional Numpy array containig image frame data.
        """

        frame = self.frame_view(frame_number)

        if self.header['NumberOfPlanes'] == 1:
            if self.color:
                return cv2.cvtColor(frame, self.header['DebayerPattern'])
            else:
                return frame
        else:
            if self.header['ColorID'] == 101:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return frame

    def read_all_frames(self):
        return [self.read_frame(idx) for idx in range(self.frame_count)]

    def correct_dynamic_range(self):
        """
        Test if the pixel values in a sample of video frames use the full dynamic range of 16bit.
        To this end, for up to ten evenly spaced frames (including the first and last frame) the
        maximal value is determined. The "shift_pixels" parameter is set to the number of bits
        which are not used in all the pixels tested. After the call to this method, calls to
        "read_frame_raw" return frame values left-shifted by this number of bits. Pixels in frames
        outside the sample which would overflow are saturated at 65535.

        :return: -
        """

        if self.frame_count == 0:
            return

        # Take a sample of up to ten evenly spaced frames.
        frame_ids = np.unique(np.linspace(0, self.frame_count - 1,
                                          min(10, self.frame_count)).astype(int))

        # Compute the maximal value of a (color) channel pixel within the sample.
        max_pixel_value = max([np.max(self.frame_view(frame_id)) for frame_id in frame_ids])
        self.frame_number = -1

        # Compute the number of unused "head room" bits. Subsequent calls to "read_frame_raw"
        # will return pixel values left-shifted by this number.
        self.shift_pixels = 16 - int(max_pixel_value).bit_length()
        # print ("shift pixels: " + str(self.shift_pixels))

        # Pixel values above "shift_limit" would overflow when shifted. In this case the shift is
        # done as a saturating multiplication with "shift_factor" (one entry per OpenCV channel).
        self.shift_limit = 65535 >> self.shift_pixels
        self.shift_factor = (float(1 << self.shift_pixels),) * 4

    def read_trailer(self):
        """
        Read the "Trailer" of SER file with time stamps in UTC for every image
        frame. Those values are "optional".

        :return:    trailer             List containing "datetime" objects for
                                        time stamps in UTC. Otherwise "None"
        """

        self.fid.seek(self.HEADER_SIZE + self.frame_count * self.frame_size)

        content = self.fid.read()

        self.frame_number = self.frame_count

        # The trailer is missing or incomplete (e.g. truncated capture).
        if self.frame_count < self.header['FrameCount'] or not content or \
                len(content) < 8 * self.frame_count:
            return None

        return [datetime.datetime(1, 1, 1) + datetime.timedelta(
            microseconds=value // 10) for value in struct.unpack(
                    '<{0}Q'.format(self.frame_count), content[:8 * self.frame_count])]

    def release(self):
        """
        Release the memory map and close the file. If frame views handed out earlier are still
        alive, the map cannot be closed explicitly. It is then unmapped when the last view is
        garbage collected.

        :return: -
        """

        self.frame_data = None
        if self.mmap is not None:
            try:
                self.mmap.close()
            except BufferError:
                pass
            self.mmap = None
        self.fid.close()

    def close(self):
        self.release()


if __name__ == "__main__":

    import ser_parser
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # file_path = r'E:\SW-Development\Python\PlanetarySystemStacker\Examples\SER_Chris-Garry' \
    #             r'\SER_GRAYSCALED_12bit_BigEndian_352_400.ser'
    # file_path = r'E:\SW-Development\Python\PlanetarySystemStacker\Examples\SER_Chris-Garry' \
    #             r'\SER_GRAYSCALED_16bit_LittleEndian_397_397.ser'
    # file_path = r'E:\SW-Development\Python\PlanetarySystemStacker\Examples\SER_Chris-Garry' \
    #             r'\SER_RGGB_16bit_LittleEndian_397_397.ser'
    # file_path = r'E:\SW-Development\Python\PlanetarySystemStacker\Examples\Sun_LauraMS' \
    #               r'\LauraMS_AR12680_2017-09-17_T_11-44-23-0221_SolarContinuum.ser'
    file_path = r'D:\SW-Development\Python\PlanetarySystemStacker\Examples\SER_Steffen-Elste\garten_1024x768-000001__16-18-36__data.ser'

    cap = ser_parser.SERParser(file_path)
    if cap.warn_message is not None:
        print (cap.warn_message)

    last_frame_read = cap.read_frame(3)
    frame_count = cap.frame_count
    shape = last_frame_read.shape
    color = cap.color
    dtype = cap.PixelDepthPerPlane
    all_frames = cap.read_all_frames()

    # Check the OpenCV BGR and Matplotlibs RGB color orders
    if color:
        plt.imshow(last_frame_read)
    else:
        plt.imshow(last_frame_read, cmap='gray')
    plt.show()
