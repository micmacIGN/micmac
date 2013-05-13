#include "Cloud.h"

#include <fstream>
#include <iostream>
//#include <cstdlib>

#include <QFileInfo>

using namespace std;
using namespace Cloud_;

#include "d:\culture3D\include\poisson\ply.h"

static PlyProperty colored_vert_props[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,x), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,y), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,z), 0, 0, 0, 0},
    {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex,red), 0, 0, 0, 0},
    {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex,green), 0, 0, 0, 0},
    {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex,blue), 0, 0, 0, 0},
    {"alpha", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex,alpha), 0, 0, 0, 0}
};

/*static PlyProperty colored_vert_props[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,x), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,y), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,z), 0, 0, 0, 0},
    {"r",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,r), 0, 0, 0, 0},
    {"g",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,g), 0, 0, 0, 0},
    {"b",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,b), 0, 0, 0, 0},
    {"a",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyColoredVertex,a), 0, 0, 0, 0}
};*/

static PlyProperty oriented_vert_props[] = {
    {"x",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,x ), 0, 0, 0, 0},
    {"y",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,y ), 0, 0, 0, 0},
    {"z",  PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,z ), 0, 0, 0, 0},
    {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,nx), 0, 0, 0, 0},
    {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,ny), 0, 0, 0, 0},
    {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex,nz), 0, 0, 0, 0}
};

Pt3D::Pt3D(){}

Pt3D::Pt3D(float x, float y, float z)
{
    m_x = x; m_y = y; m_z = z;
}

/*!
    Assigns a copy of \a Pt3D to this Pt3D, and returns a reference to it.
*/
Pt3D &Pt3D::operator=(const Pt3D &pt)
{
    m_x = pt.m_x;
    m_y = pt.m_y;
    m_z = pt.m_z;

    return *this;
}

Vertex::Vertex(Pt3D pos, QColor col)
{
    m_position = pos;
    m_color = col;
}

/*!
    Read a ply file, store the point cloud and returns true if success.
*/
bool Cloud::loadPly( const string &i_filename )
{
    PlyFile * thePlyFile;

    int nelems;
    char **elist;
    int file_type;
    float version;
    int nprops;
    int num_elems;
    char *elem_name;
    PlyProperty **plist=NULL;
    sPlyColoredVertex **vlist=NULL;

    thePlyFile = ply_open_for_reading( const_cast<char *>(i_filename.c_str()), &nelems, &elist, &file_type, &version);

    elem_name = elist[0];
    plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

    #ifdef _DEBUG
        cout << "file "		<< i_filename	<< endl;
        cout << "version "	<< version		<< endl;
        cout << "type "		<< file_type	<< endl;
        cout << "nb elem "	<< nelems		<< endl;
        cout << "num elem "	<< num_elems	<< endl;
    #endif

    for (int i = 0; i < nelems; i++)
    {
        // get the description of the first element
        elem_name = elist[i];
        plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

        // print the name of the element, for debugging
        #ifdef _DEBUG
            printf ("element %s %d\n", elem_name, num_elems);
        #endif

        if (equal_strings ("vertex", elem_name))
        {
            // create a vertex list to hold all the vertices
            vlist = (sPlyColoredVertex **) malloc (sizeof (sPlyColoredVertex *) * num_elems);

            // set up for getting vertex elements
            for (int j = 0; j < 7 ;++j)
                ply_get_property (thePlyFile, elem_name, &colored_vert_props[j]);

            // grab all the vertex elements
            for (int j = 0; j < num_elems; j++)
            {
                // grab an element from the file
                vlist[j] = (sPlyColoredVertex *) malloc (sizeof (sPlyColoredVertex));

                ply_get_element_setup(thePlyFile,elem_name,7,colored_vert_props);
                ply_get_element (thePlyFile, (void *) vlist[j]);

                //printf ("vertex: %g %g %g %u %u %u\n", vlist[j]->x, vlist[j]->y, vlist[j]->z, vlist[j]->r, vlist[j]->g, vlist[j]->b);
                printf ("vertex: %g %g %g %u %u %u\n", vlist[j]->x, vlist[j]->y, vlist[j]->z, vlist[j]->red, vlist[j]->green, vlist[j]->blue);

                Pt3D thePt( vlist[j]->x, vlist[j]->y, vlist[j]->z );
                QColor theColor( vlist[j]->red, vlist[j]->green, vlist[j]->blue );

                addVertex( Vertex (thePt, theColor) );

            }
        }
    }

    #ifdef _DEBUG
        cout << "nombre de points dans le cloud: " << getVertexNumber() << endl;
    #endif

    ply_close (thePlyFile);

    return true;
}

void Cloud::addVertex(const Vertex &vert)
{
    m_vertices.push_back(vert);
}

int Cloud::getVertexNumber()
{
    return m_vertices.size();
}

Vertex Cloud::getVertex(unsigned int nb_vert)
{
    if (m_vertices.size() > nb_vert)
    {
        return m_vertices[nb_vert];
    }
    else
    {
        cout << "error accessing point cloud vector in Cloud::getVertex" << endl;
    }
}

