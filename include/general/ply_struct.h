#ifndef __PLY_STRUCT_H__
#define __PLY_STRUCT_H__

#define PLY_START_TYPE 0
#define PLY_CHAR       1
#define PLY_SHORT      2
#define PLY_INT        3
#define PLY_UCHAR      4
#define PLY_USHORT     5
#define PLY_UINT       6
#define PLY_FLOAT      7
#define PLY_DOUBLE     8
#define PLY_INT_8      9
#define PLY_UINT_8     10
#define PLY_INT_16     11
#define PLY_UINT_16    12
#define PLY_INT_32     13
#define PLY_UINT_32    14
#define PLY_FLOAT_32   15
#define PLY_FLOAT_64   16

#define PLY_END_TYPE   17

typedef struct sVertex
{
    float x,y,z;             /* the usual 3-space position of a vertex */
} sVertex;

typedef struct sVertex64
{
    double x,y,z;
} sVertex64;

typedef struct sPlyOrientedVertex
{
    float x, y, z, nx, ny, nz;
} sPlyOrientedVertex;

typedef struct sPlyOrientedVertex64
{
    double x, y, z;
    float nx, ny, nz;
} sPlyOrientedVertex64;

typedef struct sFace
{
    unsigned char nverts;    /* number of vertex indices in list */
    int *verts;              /* vertex index list */
} sFace;

typedef struct ElPlyFace
{
    unsigned char nr_vertices;
    int *vertices;
    int segment;
} ElPlyFace;

typedef struct sPlyColoredVertex
{
    float x, y, z;
    unsigned char red, green, blue;
} sPlyColoredVertex;

typedef struct sPlyColoredVertex64
{
    double x, y, z;
    unsigned char red, green, blue;
} sPlyColoredVertex64;


typedef struct sPlyColoredVertexWithAlpha
{
    float x, y, z;
    unsigned char red, green, blue, alpha;
} sPlyColoredVertexWithAlpha;

typedef struct sPlyColoredVertexWithAlpha64
{
    double x, y, z;
    unsigned char red, green, blue, alpha;
} sPlyColoredVertexWithAlpha64;


typedef struct sPlyOrientedColoredVertex
{
    float x, y, z, nx, ny, nz;
    unsigned char red, green, blue;
} sPlyOrientedColoredVertex;

typedef struct sPlyOrientedColoredVertex64
{
    double x, y, z;
    float nx, ny, nz;
    unsigned char red, green, blue;
} sPlyOrientedColoredVertex64;

typedef struct sPlyOrientedColoredAlphaVertex
{
    float x, y, z, nx, ny, nz;
    unsigned char red, green, blue, alpha;
} sPlyOrientedColoredAlphaVertex;

typedef struct sPlyOrientedColoredAlphaVertex64
{
    double x, y, z;
    float nx, ny, nz;
    unsigned char red, green, blue, alpha;
} sPlyOrientedColoredAlphaVertex64;

#endif
