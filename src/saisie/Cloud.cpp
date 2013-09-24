#include "Cloud.h"

#include <fstream>
#include <iostream>

#include "poisson/ply.h"


using namespace std;
using namespace Cloud_;

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

Vertex::Vertex()
{
    m_position = Pt3dr(0.,0.,0.);
    m_color    = QColor();
    m_bVisible  = true;
}

Vertex::Vertex(Pt3dr pos, QColor col)
{
    m_position  = pos;
    m_color     = col;
    m_bVisible  = true;
}

/*!
    Read a ply file, store the point cloud
*/
Cloud* Cloud::loadPly(string i_filename ,int* incre)
{
    vector <Vertex> ptList;

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

                        ptList.push_back( Vertex (Pt3dr ( vertex->x, vertex->y, vertex->z ), QColor( vertex->red, vertex->green, vertex->blue, vertex->alpha )));
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

                            ptList.push_back( Vertex (Pt3dr ( vertex->x, vertex->y, vertex->z ), QColor( vertex->red, vertex->green, vertex->blue )));
                        }
                    }
                    else
                    {
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

                            ptList.push_back( Vertex (Pt3dr ( vertex->x, vertex->y, vertex->z ), QColor(Red, Green, Blue )));
                        }
                    }

                    break;
                }
            case 3:
                {
                    // set up for getting vertex elements
                    for (int j = 0; j < nprops ;++j)
                        ply_get_property (thePlyFile, elem_name, &vert_props[j]);

                    sVertex *vertex = (sVertex *) malloc (sizeof (sVertex));

                    // grab all the vertex elements
                    for (int j = 0; j < num_elems; j++)
                    {
                        if (incre) *incre = 100.0f*(float)j/num_elems;

                        // grab an element from the file
                        ply_get_element (thePlyFile, (void *) vertex);

                        #ifdef _DEBUG
                            printf ("vertex: %g %g %g\n", vertex->x, vertex->y, vertex->z);
                        #endif

                        ptList.push_back( Vertex (Pt3dr ( vertex->x, vertex->y, vertex->z ), QColor( 255, 255, 255 )));
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
        printf("verification - nombre de points dans le nuage: %d\n", (int) ptList.size() );
    #endif

    ply_close (thePlyFile);

    if(incre) *incre = 0;

    return new Cloud(ptList);
}

void Cloud::addVertex(const Vertex &vert)
{
    m_vertices.push_back(vert);
}

int Cloud::size()
{
    return m_vertices.size();
}

Vertex& Cloud::getVertex(uint nb_vert)
{
    if (m_vertices.size() > nb_vert)
    {
        return m_vertices[nb_vert];
    }
    else
    {
        printf("error accessing point cloud vector in Cloud::getVertex");
    }

    return m_vertices[0];
}

void Cloud::clear()
{
    m_vertices.clear();
}

Cloud::Cloud()
    :
      m_translation(),
      m_scale(0.)
{}

Cloud::Cloud(vector<Vertex> const & vVertex)
{
    for (uint aK=0; aK< vVertex.size(); aK++)
    {
        addVertex(vVertex[aK]);
    }

    m_translation = Pt3dr(0.,0.,0.);
    m_scale = 0.f;
}

