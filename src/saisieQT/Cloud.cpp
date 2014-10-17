#include "Cloud.h"

typedef struct sVertex
{
    float x,y,z;             /* the usual 3-space position of a vertex */
} sVertex;
typedef struct sPlyOrientedVertex
{
    float x, y, z, nx, ny, nz;
} sPlyOrientedVertex;
typedef struct sPlyColoredVertex
{
    float x, y, z;
    unsigned char red, green, blue;
} sPlyColoredVertex;

typedef struct sPlyColoredVertexWithAlpha
{
    float x, y, z;
    unsigned char red, green, blue, alpha;
} sPlyColoredVertexWithAlpha;

static PlyProperty vert_props[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertexWithAlpha,x), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertexWithAlpha,y), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertexWithAlpha,z), 0, 0, 0, 0},
};

static PlyProperty colored_a_vert_props[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertexWithAlpha,x), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertexWithAlpha,y), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertexWithAlpha,z), 0, 0, 0, 0},
    {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha,red), 0, 0, 0, 0},
    {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha,green), 0, 0, 0, 0},
    {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha,blue), 0, 0, 0, 0},
    {"alpha", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha,alpha), 0, 0, 0, 0}
};

static PlyProperty colored_vert_props[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,x), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,y), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,z), 0, 0, 0, 0},
    {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex,red), 0, 0, 0, 0},
    {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex,green), 0, 0, 0, 0},
    {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex,blue), 0, 0, 0, 0},
};

static PlyProperty oriented_vert_props[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,x ), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,y ), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,z ), 0, 0, 0, 0},
    {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,nx), 0, 0, 0, 0},
    {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,ny), 0, 0, 0, 0},
    {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,nz), 0, 0, 0, 0}
};

/*!
    Read a ply file, store the point cloud
*/
GlCloud* GlCloud::loadPly(string i_filename ,int* incre)
{
    int type = 0;
    vector <GlVertex> ptList;

    PlyFile * thePlyFile;

    int nelems;
    char **elist;
    int file_type;
    float version;
    int nprops;
    int num_elems;
    char *elem_name;
    PlyProperty ** plist = NULL;
    (void)plist;

    thePlyFile = ply_open_for_reading( const_cast<char *>(i_filename.c_str()), &nelems, &elist, &file_type, &version);

    elem_name = elist[0];
    plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

    #ifdef _DEBUG
        printf ("file %s\n"    , i_filename.c_str());
        printf ("version %f\n" , version);
        printf ("type %d\n"	   , file_type);
        printf ("nb elem %d\n" , nelems);
        printf ("num elems %d\n", num_elems);
    #endif

    for (int i = 0; i < nelems; i++)
    {
        // get the description of the first element
        elem_name = elist[i];
        plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

        // print the name of the element, for debugging
        #ifdef _DEBUG
            printf ("element %s %d %d\n", elem_name, num_elems, nprops);
        #endif

        if (equal_strings ("vertex", elem_name))
        {
            switch(nprops)
            {
            case 7:
                {
                    type = 2;
                    // setup for getting vertex elements
                    for (int j = 0; j < nprops ;++j)
                        ply_get_property (thePlyFile, elem_name, &colored_a_vert_props[j]);

                    sPlyColoredVertexWithAlpha * vertex = (sPlyColoredVertexWithAlpha *) malloc (sizeof (sPlyColoredVertexWithAlpha));

                    // grab all the vertex elements
                    for (int j = 0; j < num_elems; j++)
                    {
                        if (incre) *incre = 100.0f*(float)j/num_elems;

                        // grab an element from the file
                        ply_get_element (thePlyFile, (void *) vertex);

                        #ifdef _DEBUG
                            printf ("vertex--: %g %g %g %u %u %u %u\n", vertex->x, vertex->y, vertex->z, vertex->red, vertex->green, vertex->blue, vertex->alpha);
                        #endif

                        ptList.push_back( GlVertex (Pt3dr ( vertex->x, vertex->y, vertex->z ), QColor( vertex->red, vertex->green, vertex->blue, vertex->alpha )));
                    }
                    break;
                }

            case 6:
                {
                    // can be (x y z r g b) or (x y z nx ny nz)
                    bool wNormales = false;
                    PlyElement *elem = NULL;

                    for (int i = 0; i < nelems; i++)
                        if (equal_strings ("vertex", thePlyFile->elems[i]->name))
                            elem = thePlyFile->elems[i];

                    for (int i = 0; i < nprops; i++)
                        if ( "nx"==elem->props[i]->name )   wNormales = true;

                    if (!wNormales)
                    {
                        type = 1;
                        for (int j = 0; j < nprops ;++j)
                            ply_get_property (thePlyFile, elem_name, &colored_vert_props[j]);

                        sPlyColoredVertex *vertex = (sPlyColoredVertex *) malloc (sizeof (sPlyColoredVertex));

                        for (int j = 0; j < num_elems; j++)
                        {
                            if (incre) *incre = 100.0f*(float)j/num_elems;

                            ply_get_element (thePlyFile, (void *) vertex);

                            #ifdef _DEBUG
                                printf ("vertex: %g %g %g %u %u %u\n", vertex->x, vertex->y, vertex->z, vertex->red, vertex->green, vertex->blue);
                            #endif

                            ptList.push_back( GlVertex (Pt3dr ( vertex->x, vertex->y, vertex->z ), QColor( vertex->red, vertex->green, vertex->blue )));
                        }
                    }
                    else
                    {
                        type = 3;
                        for (int j = 0; j < nprops ;++j)
                            ply_get_property (thePlyFile, elem_name, &oriented_vert_props[j]);

                        sPlyOrientedVertex *vertex = (sPlyOrientedVertex *) malloc (sizeof (sPlyOrientedVertex));

                        for (int j = 0; j < num_elems; j++)
                        {
                            if (incre) *incre = 100.0f*(float)j/num_elems;

                            ply_get_element (thePlyFile, (void *) vertex);

                            #ifdef _DEBUG
                                printf ("vertex: %g %g %g %g %g %g\n", vertex->x, vertex->y, vertex->z, vertex->nx, vertex->ny, vertex->nz);
                            #endif

                            int Red   = (int) ((vertex->nx + 1.f)*122.5);
                            int Green = (int) ((vertex->ny + 1.f)*122.5);
                            int Blue  = (int) ((vertex->nz + 1.f)*122.5);

                            /*if (Red > 255)   cout << "Red= " << Red << endl;
                            if (Green > 255) cout << "Green= " << Green << endl;
                            if (Blue >255)   cout << "Blue= " << Blue << endl;*/

                            ptList.push_back( GlVertex (Pt3dr ( vertex->x, vertex->y, vertex->z ), QColor(Red, Green, Blue )));
                        }
                    }
                    break;
                }
            case 3:
                {
                    for (int j = 0; j < nprops ;++j)
                        ply_get_property (thePlyFile, elem_name, &vert_props[j]);

                    sVertex *vertex = (sVertex *) malloc (sizeof (sVertex));

                    for (int j = 0; j < num_elems; j++)
                    {
                        if (incre) *incre = 100.0f*(float)j/num_elems;

                        ply_get_element (thePlyFile, (void *) vertex);

                        #ifdef _DEBUG
                            printf ("vertex: %g %g %g\n", vertex->x, vertex->y, vertex->z);
                        #endif

                        ptList.push_back( GlVertex (Pt3dr ( vertex->x, vertex->y, vertex->z ), QColor( 255, 255, 255 )));
                }
                break;
                }
             default:
                {
                    printf("unable to load a ply unless number of properties is 3, 6 or 7\n");
                    break;
                }
            }
        }
    }

    #ifdef _DEBUG
        printf("check - point number in cloud: %d\n", (int) ptList.size() );
    #endif

    ply_close (thePlyFile);

    if(incre) *incre = 0;

    return new GlCloud(ptList, type);
}

void GlCloud::addVertex(const GlVertex &vertex)
{
    _vertices.push_back(vertex);
}

int GlCloud::size()
{
    return _vertices.size();
}

GlVertex& GlCloud::getVertex(uint nb_vert)
{
    if (_vertices.size() > nb_vert)
    {
        return _vertices[nb_vert];
    }
    else
    {
        printf("error accessing point cloud vector in Cloud::getVertex");
    }

    return _vertices[0];
}

void GlCloud::clear()
{
    _vertices.clear();
}

GlCloud::GlCloud(vector<GlVertex> const & vVertex, int type):
    _type(type)
{
    for (uint aK=0; aK< vVertex.size(); aK++)
    {
        addVertex(vVertex[aK]);
    }

    _position = Pt3dr(0.,0.,0.);
    _scale = Pt3dr(1.f,1.f,1.f);
}

void GlCloud::draw()
{
    //glEnable(GL_DEPTH_TEST);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    _vertexbuffer.bind();
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    _vertexbuffer.release();

    _vertexColor.bind();
    glColorPointer(3, GL_FLOAT, 0, NULL);
    _vertexColor.release();

    glDrawArrays( GL_POINTS, 0, size()*3 );

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

   // glDisable(GL_DEPTH_TEST);
}

void GlCloud::setBufferGl(bool onlyColor)
{
    if(_vertexbuffer.isCreated() && !onlyColor)
        _vertexbuffer.destroy();
    if(_vertexColor.isCreated())
        _vertexColor.destroy();

    uint sizeCloud = size();
    GLfloat* vertices = NULL, *colors = NULL;

    if(!onlyColor)
        vertices = new GLfloat[sizeCloud*3];

    colors     = new GLfloat[sizeCloud*3];

    for(uint bK=0; bK< sizeCloud; bK++)
    {
        GlVertex vert = getVertex(bK);
        Pt3dr  pos  = vert.getPosition();
        QColor colo = vert.getColor();
        if(!onlyColor)
        {
            vertices[bK*3 + 0 ] = pos.x;
            vertices[bK*3 + 1 ] = pos.y;
            vertices[bK*3 + 2 ] = pos.z;
        }
        if(vert.isVisible())
        {
            colors[bK*3 + 0 ]   = colo.redF();
            colors[bK*3 + 1 ]   = colo.greenF();
            colors[bK*3 + 2 ]   = colo.blueF();
        }
        else
        {
            colors[bK*3 + 0 ]   = colo.redF()   *0.7;
            colors[bK*3 + 1 ]   = colo.greenF() *0.6;
            colors[bK*3 + 2 ]   = colo.blueF()  *0.8;
        }
    }

    if(!onlyColor)
    {
        _vertexbuffer.create();
        _vertexbuffer.setUsagePattern(QGLBuffer::StaticDraw);
        _vertexbuffer.bind();
        _vertexbuffer.allocate(vertices, sizeCloud* 3 * sizeof(GLfloat));
        _vertexbuffer.release();
    }

    _vertexColor.create();
    _vertexColor.setUsagePattern(QGLBuffer::StaticDraw);
    _vertexColor.bind();
    _vertexColor.allocate(colors, sizeCloud* 3 * sizeof(GLfloat));
    _vertexColor.release();

    if(!onlyColor)
        delete [] vertices;
    delete [] colors;
}

