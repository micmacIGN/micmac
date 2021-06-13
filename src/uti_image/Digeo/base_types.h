#ifndef __BASE_TYPES_DEFINITIONS__
#define __BASE_TYPES_DEFINITIONS__

#include <stdint.h>

#define U_INT1 uint8_t
#define U_INT2 uint16_t
#define U_INT4 uint32_t
#define U_INT8 uint64_t
#define INT1   int8_t
#define INT4   int32_t
#define INT2   int16_t
#define _INT8  int64_t

#define REAL4 float
#define REAL8 double
#define REAL16 long double

#define INTByte8 int64_t

#ifndef INT
#define INT INT4
#endif

#ifndef U_INT
#define U_INT unsigned int
#endif

#ifndef REAL
#define REAL   REAL8
#endif

#endif
