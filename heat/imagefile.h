#ifndef __FSUTIL_IMAGEFILE_H__
#define __FSUTIL_IMAGEFILE_H__

#include <memory.h>
#include <fstream>



class ImageFile {

  //**** MEMBER VARIABLES ****
  unsigned char *_red;
  unsigned char *_green;
  unsigned char *_blue;

  int _width;
  int _height;

public:

  //**** MANAGERS ****
  ImageFile() { _red = 0; _green = 0; _blue = 0; _width = _height = 0; }
  ~ImageFile() { delete[] _red; delete[] _green; delete[] _blue; }

  //**** PUBLIC INTERFACE ****
  void clear(unsigned char r=0, unsigned char g=0, unsigned char b=0);

  void allocate(int width, int height);

  unsigned char &red(int x, int y) { return _red[y * _width + x]; }
  const unsigned char &red(int x, int y) const { return _red[y * _width + x]; }

  unsigned char &green(int x, int y) { return _green[y * _width + x]; }
  const unsigned char &green(int x, int y) const { return _green[y * _width + x]; }

  unsigned char &blue(int x, int y) { return _blue[y * _width + x]; }
  const unsigned char &blue(int x, int y) const { return _blue[y * _width + x]; }

  void set_rgb(int x, int y, unsigned char r, unsigned char g, unsigned char b) { red(x,y) = r; green(x,y) = g; blue(x,y) = b; }

  bool write_ppm(const char *filename);
  bool read_ppm(const char *filename);
};




void 
ImageFile::clear(unsigned char r, unsigned char g, unsigned char b)
{
  memset(_red, r, _width * _height);
  memset(_green, g, _width * _height);
  memset(_blue, b, _width * _height);
}

void 
ImageFile::allocate(int width, int height)
{
  if (_red) delete[] _red;
  if (_green) delete[] _green;
  if (_blue) delete[] _blue;

  _red = new unsigned char[width * height];
  _green = new unsigned char[width * height];
  _blue = new unsigned char[width * height];

  _width = width;
  _height = height;
}

bool 
ImageFile::write_ppm(const char *filename)
{
  std::ofstream file;
  file.open(filename, std::ios::binary);

  if (!file) {
    printf("[ERROR] ImageFile::WritePPM - could not open file %s\n", filename);
    return false;
  }

  file << "P6 ";

  file << _width << " ";
  file << _height << " ";
  file << 255 << "\n";

  unsigned char *rptr = _red;
  unsigned char *gptr = _green;
  unsigned char *bptr = _blue;

  const unsigned char *rlast = rptr + (_width * _height);

  while (rptr != rlast) {
    file << *rptr++;
    file << *gptr++;
    file << *bptr++;
  }

  return true;
}


bool 
ImageFile::read_ppm(const char *filename)
{
  return false;
}




#endif


