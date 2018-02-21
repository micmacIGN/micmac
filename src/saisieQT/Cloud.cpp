#include "Cloud.h"
#include "general/PlyFile.h"
/*!
    Read a ply file, store the point cloud
*/

GlCloud* GlCloud::loadPly(string i_filename)
{
    int type = 0;
    vector <GlVertex> ptList;

    int nelems;
    char **elist;
    int file_type;
    float version;
    int nprops;
    int num_elems;
    char *elem_name;

    PlyFile *thePlyFile = ply_open_for_reading( const_cast<char *>(i_filename.c_str()), &nelems, &elist, &file_type, &version);

    elem_name = elist[0];
    ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

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
        PlyProperty **plist = NULL;
        plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);

        // print the name of the element, for debugging
        #ifdef _DEBUG
            printf ("element %s %d %d\n", elem_name, num_elems, nprops);
        #endif

        if (equal_strings ("vertex", elem_name))
        {
            int external_type = plist[0]->external_type;
            std::cout << "external_type : "<<external_type<<std::endl;
            
            switch(nprops)
            {
            case 10: // x y z nx ny nz r g b a
                {
                    type = 5;
                    PlyProperty props[] = {
                        {"x",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredAlphaVertex64,x ), 0, 0, 0, 0},
                        {"y",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredAlphaVertex64,y ), 0, 0, 0, 0},
                        {"z",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredAlphaVertex64,z ), 0, 0, 0, 0},
                        {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredAlphaVertex64,nx), 0, 0, 0, 0},
                        {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredAlphaVertex64,ny), 0, 0, 0, 0},
                        {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredAlphaVertex64,nz), 0, 0, 0, 0},
                        {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredAlphaVertex64,red), 0, 0, 0, 0},
                        {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredAlphaVertex64,green), 0, 0, 0, 0},
                        {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredAlphaVertex64,blue), 0, 0, 0, 0},
                        {"alpha",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredAlphaVertex64,alpha), 0, 0, 0, 0}
                    };
                    
                    for(int j=0;j<nprops;++j)
                    {
                        ply_get_property(thePlyFile, elem_name, &props[j]);
                    }

                    // grab all the vertex elements
                    for (int j = 0; j < num_elems; j++)
                    {
                        sPlyOrientedColoredAlphaVertex64 vertex;
                        ply_get_element (thePlyFile, (void *) &vertex);

#ifdef _DEBUG
    printf ("vertex--: %g %g %g %g %g %g %u %u %u %u\n", vertex.x, vertex.y, vertex.z, vertex.nx, vertex.ny, vertex.nz, vertex.red, vertex.green, vertex.blue, vertex.alpha);
#endif

						ptList.push_back( GlVertex (QVector3D ( vertex.x, vertex.y, vertex.z ), QColor( vertex.red, vertex.green, vertex.blue, vertex.alpha ), QVector3D(vertex.nx, vertex.ny, vertex.nz)));
                    }
                    break;
                }
            case 9: // x y z nx ny nz r g b
                {
                    type = 4;
                    PlyProperty props[] = {
                        {"x",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredVertex64,x ), 0, 0, 0, 0},
                        {"y",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredVertex64,y ), 0, 0, 0, 0},
                        {"z",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredVertex64,z ), 0, 0, 0, 0},
                        {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredVertex64,nx), 0, 0, 0, 0},
                        {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredVertex64,ny), 0, 0, 0, 0},
                        {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredVertex64,nz), 0, 0, 0, 0},
                        {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredVertex64,red), 0, 0, 0, 0},
                        {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredVertex64,green), 0, 0, 0, 0},
                        {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredVertex64,blue), 0, 0, 0, 0}
                    };
                    
                    for(int j=0;j<nprops;++j)
                    {
                        ply_get_property(thePlyFile, elem_name, &props[j]);
                    }

                    // grab all the vertex elements
                    for (int j = 0; j < num_elems; j++)
                    {
                        sPlyOrientedColoredVertex64 vertex;
                        ply_get_element (thePlyFile, (void *) &vertex);

#ifdef _DEBUG
    printf ("vertex--: %g %g %g %g %g %g %u %u %u\n", vertex.x, vertex.y, vertex.z, vertex.nx, vertex.ny, vertex.nz, vertex.red, vertex.green, vertex.blue);
#endif

						ptList.push_back( GlVertex (QVector3D ( vertex.x, vertex.y, vertex.z ), QColor( vertex.red, vertex.green, vertex.blue ), QVector3D(vertex.nx, vertex.ny, vertex.nz)));
                    }
                    break;
                }
            case 7:
                {
                    type = 2;
                    PlyProperty props[] = {
                        {"x",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertexWithAlpha64,x ), 0, 0, 0, 0},
                        {"y",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertexWithAlpha64,y ), 0, 0, 0, 0},
                        {"z",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertexWithAlpha64,z ), 0, 0, 0, 0},
                        {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha64,red), 0, 0, 0, 0},
                        {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha64,green), 0, 0, 0, 0},
                        {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha64,blue), 0, 0, 0, 0},
                        {"alpha",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha64,alpha), 0, 0, 0, 0}
                    };
                    
                    for(int j=0;j<nprops;++j)
                    {
                        ply_get_property(thePlyFile, elem_name, &props[j]);
                    }

                    // grab all the vertex elements
                    for (int j = 0; j < num_elems; j++)
                    {
                        // grab an element from the file
                        sPlyColoredVertexWithAlpha64 vertex;
                        ply_get_element (thePlyFile, (void *) &vertex);

                        #ifdef _DEBUG
                            printf ("vertex--: %g %g %g %u %u %u %u\n", vertex.x, vertex.y, vertex.z, vertex.red, vertex.green, vertex.blue, vertex.alpha);
                        #endif

						ptList.push_back( GlVertex (QVector3D ( vertex.x, vertex.y, vertex.z ), QColor( vertex.red, vertex.green, vertex.blue, vertex.alpha )));
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
                        PlyProperty props[] = {
                            {"x",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertex64,x ), 0, 0, 0, 0},
                            {"y",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertex64,y ), 0, 0, 0, 0},
                            {"z",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertex64,z ), 0, 0, 0, 0},
                            {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex64,red), 0, 0, 0, 0},
                            {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex64,green), 0, 0, 0, 0},
                            {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex64,blue), 0, 0, 0, 0}
                        };
                        
                        for(int j=0;j<nprops;++j)
                        {
                            ply_get_property(thePlyFile, elem_name, &props[j]);
                        }

                        for (int j = 0; j < num_elems; j++)
                        {
                            sPlyColoredVertex64 vertex;
                            ply_get_element (thePlyFile, (void *) &vertex);

                            #ifdef _DEBUG
                                printf ("vertex: %g %g %g %u %u %u\n", vertex.x, vertex.y, vertex.z, vertex.red, vertex.green, vertex.blue);
                            #endif

							ptList.push_back( GlVertex (QVector3D ( vertex.x, vertex.y, vertex.z ), QColor( vertex.red, vertex.green, vertex.blue )));
                        }
                    }
                    else
                    {
                        type = 3;
                        PlyProperty props[] = {
                            {"x",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedVertex64,x ), 0, 0, 0, 0},
                            {"y",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedVertex64,y ), 0, 0, 0, 0},
                            {"z",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedVertex64,z ), 0, 0, 0, 0},
                            {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex64,nx), 0, 0, 0, 0},
                            {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex64,ny), 0, 0, 0, 0},
                            {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex64,nz), 0, 0, 0, 0}
                        };
                        
                        for(int j=0;j<nprops;++j)
                        {
                            ply_get_property(thePlyFile, elem_name, &props[j]);
                        }

                        for (int j = 0; j < num_elems; j++)
                        {
                            sPlyOrientedVertex64 vertex;
                            ply_get_element (thePlyFile, (void *) &vertex);

                            #ifdef _DEBUG
                                printf ("vertex: %g %g %g %g %g %g\n", vertex.x, vertex.y, vertex.z, vertex.nx, vertex.ny, vertex.nz);
                            #endif

							ptList.push_back( GlVertex (QVector3D ( vertex.x, vertex.y, vertex.z ), Qt::white, QVector3D(vertex.nx, vertex.ny, vertex.nz)));
                        }
                    }
                    break;
                }
            case 3:
                {
                    PlyProperty props[] = {
                        {"x",  external_type, PLY_DOUBLE, offsetof(sVertex64,x ), 0, 0, 0, 0},
                        {"y",  external_type, PLY_DOUBLE, offsetof(sVertex64,y ), 0, 0, 0, 0},
                        {"z",  external_type, PLY_DOUBLE, offsetof(sVertex64,z ), 0, 0, 0, 0}
                    };
                    
                    for(int j=0;j<nprops;++j)
                    {
                        ply_get_property(thePlyFile, elem_name, &props[j]);
                    }
                    
                    for (int j = 0; j < num_elems; j++)
                    {
                        sVertex64 vertex;
                        ply_get_element (thePlyFile, (void *) &vertex);

    #ifdef _DEBUG
                        printf ("vertex: %g %g %g\n", vertex.x, vertex.y, vertex.z);
    #endif

						ptList.push_back( GlVertex (QVector3D ( vertex.x, vertex.y, vertex.z )));
                    }
                    break;
                }
             default:
                {
                    printf("unable to load a ply unless number of properties is 3, 6, 7, 9 or 10\n");
                    break;
                }
            }
        }
        
        std::cout << "Fin"<<std::endl;
        
        if (plist!=NULL)
        {
            for(int j=0;j<nprops;++j)
            {
                free(plist[j]);
            }
            free(plist);

        }
    }

    #ifdef _DEBUG
        printf("check - point number in cloud: %d\n", (int) ptList.size() );
    #endif

    ply_close (thePlyFile);

    return new GlCloud(ptList, type, thePlyFile->comments, thePlyFile->num_comments);
}

void GlCloud::addVertex(const GlVertex &vertex)
{
    _sum = _sum + const_cast<GlVertex &>(vertex).getPosition();

    _vertices.push_back(vertex);
}

int GlCloud::size()
{
    return (int)_vertices.size();
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

GlCloud::GlCloud(vector<GlVertex> const & vVertex, int type, char **comments, int nbComments):
    _type(type),
    _sum(QVector3D(0.,0.,0.))
{
    for (uint aK=0; aK< vVertex.size(); aK++)
    {
        addVertex(vVertex[aK]);
    }

    _position = QVector3D(0.,0.,0.);
    _scale = QVector3D(1.f,1.f,1.f);

    for (int i=0;i<nbComments;i++)
        _comments.push_back(comments[i]);
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

    glDrawArrays( GL_POINTS, 0, size() );

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
        QVector3D  pos  = vert.getPosition();

        QColor colo = vert.getColor();
        if(!onlyColor)
        {
            vertices[bK*3 + 0 ] = pos.x();
            vertices[bK*3 + 1 ] = pos.y();
            vertices[bK*3 + 2 ] = pos.z();
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

